use std::error::Error;
use std::fmt;
use std::fs::{exists, read_to_string, remove_file, File};
use std::io::{self, BufRead, Write};
use std::mem::MaybeUninit;
use std::path::Path;
use std::sync::Arc;

use bytemuck;
use indicatif::ProgressBar;
use memmap2::{Mmap, MmapOptions};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rustc_hash::FxHashMap;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde_json::Deserializer as JsonDeserializer;
use smallvec::SmallVec;

pub const EMPTY_CELL: u8 = ' ' as u8;
pub const LENGTH: usize = 5;
const LOCK_FILE_NAME: &str = "db.lock";
const FLAG_FILE_NAME: &str = "db.flag";
const DATABASE_FILE_NAME: &str = "db.db";
const LETTERS: usize = 26;

const SQUARE_LEN: usize = LENGTH * LENGTH * std::mem::size_of::<u8>();
const PADDING_LEN: usize = SQUARE_LEN % std::mem::size_of::<u16>();
const HASH_LEN: usize = 2 * LENGTH * std::mem::size_of::<u16>();
const TOTAL_LEN: usize = SQUARE_LEN + PADDING_LEN + HASH_LEN;
const TXN_SIZE: usize = 1024 * 1024;
const BUF_LEN: usize = TOTAL_LEN * TXN_SIZE;

pub type PrefixSet = FxHashMap<SmallVec<[u8; LENGTH]>, Vec<u16>>;
pub type ErrT = Box<dyn Error + Send + Sync>;

pub struct WordDb {
    pub db: Arc<Mmap>,
    pub changes: [usize; LETTERS],
    pub count: usize,
    lock_file: File,
    db_folder: String,
}

impl WordDb {
    pub fn new(db_folder: String) -> Result<WordDb, ErrT> {
        let db_folder_string = db_folder.clone();
        let db_folder = Path::new(&db_folder);
        let lock_file = File::create(db_folder.join(LOCK_FILE_NAME))?;
        lock_file.try_lock()?;
        let (count, changes) = parse_flag_file(&db_folder_string)?;
        let db_file = File::open(db_folder.join(DATABASE_FILE_NAME)).map_err(|e| Box::new(e))?;
        let db = Arc::new(unsafe {
            MmapOptions::new()
                .huge(Some(21))
                .no_reserve_swap()
                .map(&db_file)
        }?);
        Ok(WordDb {
            db: db,
            lock_file: lock_file,
            db_folder: db_folder_string,
            count: count,
            changes: changes,
        })
    }

    pub fn get(&self, idx: usize) -> Option<WordDbEntry<'_>> {
        if idx < self.count {
            Some(WordDbEntry::deserialize(&self.db, idx * TOTAL_LEN))
        } else {
            None
        }
    }
}

impl Drop for WordDb {
    fn drop(&mut self) {
        self.lock_file.unlock().unwrap();
        remove_file(Path::new(&self.db_folder).join(LOCK_FILE_NAME)).unwrap();
    }
}

fn word_square_view(word_square: &[u8]) -> ArrayView2<'_, u8> {
    ArrayView2::from_shape((LENGTH, LENGTH), word_square).unwrap()
}

/// Reduces one copy during construction of database
pub struct WordDbEntry<'a> {
    pub word_square: &'a [u8; LENGTH * LENGTH],
    pub word_square_words: &'a [u16; 2 * LENGTH],
}

impl<'a> WordDbEntry<'a> {
    // For speed, we directly copy raw bits to and fro with very little consistency checking
    pub fn serialize(&self, target: &mut [u8], offset: usize) {
        target[offset..offset + SQUARE_LEN].copy_from_slice(self.word_square);
        target[offset + SQUARE_LEN] = 0;

        // We do not consider endianess
        let ptr: *const u8 = self.word_square_words.as_ptr().cast();
        target[offset + SQUARE_LEN + PADDING_LEN..offset + TOTAL_LEN]
            .copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, HASH_LEN) });
    }

    pub fn deserialize(source: &'a [u8], offset: usize) -> WordDbEntry<'a> {
        let square_bytes: &'a [u8; SQUARE_LEN] =
            source[offset..offset + SQUARE_LEN].try_into().unwrap();

        let hash_bytes: &'a [u8; HASH_LEN] = source
            [offset + SQUARE_LEN + PADDING_LEN..offset + TOTAL_LEN]
            .try_into()
            .unwrap();

        // Reinterprets &[u8; 20] as &[u16; 10] safely
        let word_square_words: &'a [u16; 2 * LENGTH] = bytemuck::cast_ref(hash_bytes);

        WordDbEntry {
            word_square: square_bytes,
            word_square_words,
        }
    }

    pub fn word_square_view(&self) -> ArrayView2<'_, u8> {
        word_square_view(self.word_square)
    }
}

#[derive(Debug)]
pub struct DbErr {
    pub reason: String,
}

impl DbErr {
    fn new(reason: &str) -> DbErr {
        DbErr {
            reason: String::from(reason),
        }
    }
}

impl fmt::Display for DbErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DbErr: {}", self.reason)
    }
}

impl Error for DbErr {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

pub fn read_json_with_visitor<
    T: FnMut(&[u8; LENGTH * LENGTH], u32, bool, u8) -> Result<(), ErrT>,
>(
    json_file: String,
    insert_record: &mut T,
) -> Result<(), ErrT> {
    let file = File::open(&json_file)?;
    let file_size = file.metadata()?.len();

    let record_length = (LENGTH + 3) * LENGTH + 2;
    let records = file_size.saturating_sub(1) / (record_length as u64);
    let pb = ProgressBar::new(records);

    let buf = io::BufReader::new(file);
    let mut deserializer = JsonDeserializer::from_reader(buf);

    struct WordSquareVisitor<'a, F> {
        insert_record: &'a mut F,
        pb: ProgressBar,
    }

    impl<'de, 'a, F> Visitor<'de> for WordSquareVisitor<'a, F>
    where
        F: FnMut(&[u8; LENGTH * LENGTH], u32, bool, u8) -> Result<(), ErrT>,
    {
        type Value = ();

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a JSON array of 5x5 character squares")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut db_id: u32 = 0;
            let mut last_top_left: Option<u8> = None;

            while let Some(row_strings) = seq.next_element::<[String; LENGTH]>()? {
                let mut data_vec: [MaybeUninit<u8>; LENGTH * LENGTH] =
                    [const { MaybeUninit::uninit() }; LENGTH * LENGTH];

                for (r, row) in row_strings.iter().enumerate() {
                    let chars: Vec<u8> = row.chars().map(|x| x as u8).collect();
                    if chars.len() != LENGTH {
                        return Err(de::Error::custom(format!(
                            "Row length mismatch at index {}: expected {}, got {}",
                            db_id,
                            LENGTH,
                            chars.len()
                        )));
                    }
                    for (c, &ch) in chars.iter().enumerate() {
                        data_vec[r * LENGTH + c].write(ch);
                    }
                }

                let data_vec = unsafe { std::mem::transmute::<_, [u8; LENGTH * LENGTH]>(data_vec) };

                let top_left = data_vec[0];
                let top_left_changed = match last_top_left {
                    Some(last) if top_left < last => {
                        return Err(de::Error::custom(format!(
                            "JSON is not sorted at index {}: '{}' < '{}'",
                            db_id, top_left, last
                        )));
                    }
                    Some(last) => top_left != last,
                    None => true,
                };
                last_top_left = Some(top_left);

                (self.insert_record)(&data_vec, db_id, top_left_changed, top_left)
                    .map_err(de::Error::custom)?;

                self.pb.inc(1);
                db_id += 1;
            }

            // If we went out early, drain the rest of the records
            while let Some(_) = seq.next_element::<[String; LENGTH]>()? {
                self.pb.inc(1);
            }

            Ok(())
        }
    }

    deserializer.deserialize_seq(WordSquareVisitor { insert_record, pb })?;

    Ok(())
}

fn parse_flag_file(db_folder: &String) -> Result<(usize, [usize; LETTERS]), ErrT> {
    let mut changes = [0; LETTERS];

    let flag_file = Path::new(db_folder).join(FLAG_FILE_NAME);
    let file_string = read_to_string(flag_file).unwrap();
    let mut file_iter = file_string.lines();
    let count = match file_iter.next() {
        Some(line) => line.parse()?,
        None => return Err(Box::new(DbErr::new("Invalid complete file. Empty file."))),
    };

    let mut i = 0;
    // read the changes from the file
    for line in file_iter {
        if i >= LETTERS {
            return Err(Box::new(DbErr::new(
                "Invalid complete file. Too many changes.",
            )));
        }
        changes[i] = line.parse()?;
        i += 1;
    }

    if i != LETTERS {
        return Err(Box::new(DbErr::new(
            "Invalid complete file. Not enough changes.",
        )));
    }

    return Ok((count, changes));
}

pub fn copy_json_to_database(
    json_file: String,
    db_folder: String,
    words: &Array2<u8>,
) -> Result<(), ErrT> {
    let mut changes = [0; LETTERS];
    let reverse_word_map = create_reverse_word_map(words);

    // If we have an already running process
    let lock_file = Path::new(&db_folder).join(LOCK_FILE_NAME);
    if exists(&lock_file)? {
        return Err(Box::new(DbErr::new("Lock file exists")));
    }

    // Note, there is a race condition here, but this is a personal project, so I'm not really
    // worried about it

    let lock_file = File::create(lock_file)?;
    lock_file.try_lock()?;

    // If we already have done this
    let flag_file = Path::new(&db_folder).join(FLAG_FILE_NAME);
    if exists(&flag_file)? {
        return Ok(());
    }

    let database_file = Path::new(&db_folder).join(DATABASE_FILE_NAME);
    let mut db_file_handle = File::create(database_file)?;
    let mut write_buffer: [u8; BUF_LEN] = [0; BUF_LEN];
    let mut idx: usize = 0;
    let mut count: usize = 0;

    read_json_with_visitor(json_file, &mut |data: &[u8; LENGTH * LENGTH],
                                            id,
                                            top_left_changed,
                                            top_left|
     -> Result<(), ErrT> {
        let offset = idx * TOTAL_LEN;
        let wsw = words_in_square(data, &reverse_word_map).unwrap();
        let word_db_entry = WordDbEntry {
            word_square: data,
            word_square_words: &wsw,
        };

        word_db_entry.serialize(&mut write_buffer, offset);
        idx += 1;
        count += 1;

        if idx >= TXN_SIZE {
            let num_bytes = db_file_handle.write(&write_buffer)?;
            if num_bytes != BUF_LEN {
                return Err(Box::new(DbErr::new("Could not write to disk")));
            }
            idx = 0;
        }

        if top_left_changed {
            changes[top_left as usize - 'a' as usize] = id;
        }

        Ok(())
    })?;

    // Need to finish writing to file
    db_file_handle.write(&write_buffer[0..idx * TOTAL_LEN])?;
    db_file_handle.flush()?;

    // Write the flag file
    let mut flag_file = File::create(flag_file)?;

    writeln!(flag_file, "{}", count)?;

    for i in 0..LETTERS {
        writeln!(flag_file, "{}", changes[i].to_string())?;
    }

    // remove lock file
    lock_file.unlock()?;
    remove_file(Path::new(&db_folder).join(LOCK_FILE_NAME))?;

    Ok(())
}

fn words_in_square(
    ws: &[u8],
    reverse_word_map: &FxHashMap<Array1<u8>, u16>,
) -> Result<[u16; 2 * LENGTH], ErrT> {
    let mut result = [const { MaybeUninit::uninit() }; 2 * LENGTH];
    let mut idx = 0;
    let ws = word_square_view(ws);
    for i in 0..=1 {
        for word in ws.axis_iter(Axis(i)) {
            result[idx].write(
                *reverse_word_map
                    .get(&word.to_owned())
                    .expect("word not found"),
            );
            idx += 1;
        }
    }

    assert!(idx == 2 * LENGTH);

    let mut result = unsafe { std::mem::transmute::<_, [u16; 2 * LENGTH]>(result) };
    result.sort();

    Ok(result)
}

/// Words are assumed sorted
pub fn read_words_from_file(file: String) -> Array2<u8> {
    let mut raw_data = Vec::new();
    let mut count = 0;
    if let Ok(lines) = read_lines(file) {
        for line in lines.map_while(Result::ok) {
            let line = line.trim().to_ascii_lowercase();
            if line.len() == LENGTH && line.is_ascii() {
                raw_data.extend(line.as_bytes());
                count += 1;
            }
        }
    }

    Array2::from_shape_vec((count, LENGTH), raw_data).unwrap()
}

fn create_reverse_word_map(words: &Array2<u8>) -> FxHashMap<Array1<u8>, u16> {
    let mut result = FxHashMap::default();
    for i in 0..words.shape()[0] {
        result.insert(words.slice(s![i, ..]).to_owned(), i as u16);
    }

    result
}

pub fn generate_prefixes_from_words(words: &Array2<u8>) -> PrefixSet {
    let mut result: PrefixSet = FxHashMap::default();
    for (word_id, word) in words.axis_iter(Axis(0)).enumerate() {
        for i in 0..LENGTH {
            let key_vec = word
                .slice(s![0..i])
                .to_owned()
                .into_iter()
                .collect::<SmallVec<_>>();
            let val = result.entry(key_vec).or_insert(Vec::new());
            val.push(word_id as u16);
        }
    }

    for (_, v) in result.iter_mut() {
        v.sort();
    }

    result
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
