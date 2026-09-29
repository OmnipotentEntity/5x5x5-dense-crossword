use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead};
use std::mem::size_of;
use std::path::Path;

use heed::{
    byteorder::NativeEndian, types::U32, BoxedError, BytesDecode, BytesEncode, Database, Env,
};
use indicatif::ProgressBar;
use ndarray::{s, Array, Array1, Array2, Axis};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde_json::Deserializer as JsonDeserializer;

pub const EMPTY_CELL: char = ' ';
pub const LENGTH: usize = 5;
const TXN_SIZE: u32 = 1024 * 1024;

pub type PrefixSet = HashMap<Vec<char>, HashSet<u16>>;
pub type ErrT = Box<dyn Error + Send + Sync>;

pub type Nu32 = U32<NativeEndian>;

/// The thing that we actually store inside of the database
pub struct WordDbEntry {
    pub word_square: Array2<char>,
    /// Contains indices into the words array for each words in the word square
    /// Indices are compressed to u16 to save space
    pub word_square_words: HashSet<u16>,
}

pub struct WordDbEntryCodec;

impl<'a> BytesEncode<'a> for WordDbEntryCodec {
    type EItem = WordDbEntry;

    fn bytes_encode(entry: &Self::EItem) -> Result<Cow<'a, [u8]>, BoxedError> {
        let mut result = Vec::new();
        result.append(
            &mut entry
                .word_square
                .iter()
                .map(|x| *x as u8)
                .collect::<Vec<_>>(),
        );
        result.append(
            &mut entry
                .word_square_words
                .iter()
                .map(|x| x.to_ne_bytes())
                .flatten()
                .collect::<Vec<_>>(),
        );

        Ok(Cow::Owned(result))
    }
}

impl<'a> BytesDecode<'a> for WordDbEntryCodec {
    type DItem = WordDbEntry;

    fn bytes_decode(bytes: &'a [u8]) -> Result<Self::DItem, BoxedError> {
        let ws = match bytes.get(..LENGTH * LENGTH) {
            Some(bytes) => Array::from_iter(bytes.iter().map(|x| *x as char))
                .into_shape_with_order((LENGTH, LENGTH))
                .unwrap(),
            None => return Err("invalid db entry".into()),
        };

        let wsw = match bytes.get(LENGTH * LENGTH..) {
            Some(bytes) => {
                if bytes.len() != size_of::<u16>() * 2 * LENGTH {
                    return Err("invalid db entry".into());
                };
                bytes
                    .chunks(2)
                    .map(|x| u16::from_ne_bytes([x[0], x[1]]))
                    .collect()
            }
            None => return Err("invalid db entry".into()),
        };

        Ok(WordDbEntry {
            word_square: ws,
            word_square_words: wsw,
        })
    }
}

pub fn read_json_with_visitor<T: FnMut(Array2<char>, u32, bool, char) -> Result<(), ErrT>>(
    json_file: String,
    records_to_read: Option<u64>,
    insert_record: &mut T,
) -> Result<(), ErrT> {
    let file = File::open(&json_file)?;
    let file_size = file.metadata()?.len();

    let record_length = (LENGTH + 3) * LENGTH + 2;
    let records = file_size.saturating_sub(1) / (record_length as u64);
    let pb = ProgressBar::new(records_to_read.unwrap_or(records));

    let buf = io::BufReader::new(file);
    let mut deserializer = JsonDeserializer::from_reader(buf);

    struct WordSquareVisitor<'a, F> {
        insert_record: &'a mut F,
        records_to_read: Option<u64>,
        pb: ProgressBar,
    }

    impl<'de, 'a, F> Visitor<'de> for WordSquareVisitor<'a, F>
    where
        F: FnMut(Array2<char>, u32, bool, char) -> Result<(), ErrT>,
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
            let mut last_top_left: Option<char> = None;

            while let Some(row_strings) = seq.next_element::<[String; LENGTH]>()? {
                let mut data_vec = Array2::from_elem((LENGTH, LENGTH), EMPTY_CELL);

                for (r, row) in row_strings.iter().enumerate() {
                    let chars: Vec<char> = row.chars().collect();
                    if chars.len() != LENGTH {
                        return Err(de::Error::custom(format!(
                            "Row length mismatch at index {}: expected {}, got {}",
                            db_id, LENGTH, chars.len()
                        )));
                    }
                    for (c, &ch) in chars.iter().enumerate() {
                        data_vec[(r, c)] = ch;
                    }
                }

                let top_left = data_vec[(0, 0)];
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

                (self.insert_record)(data_vec, db_id, top_left_changed, top_left)
                    .map_err(de::Error::custom)?;

                self.pb.inc(1);
                db_id += 1;

                if let Some(limit) = self.records_to_read {
                    if db_id as u64 >= limit {
                        break;
                    }
                }
            }
           
            // If we went out early, drain the rest of the records
            while let Some(_) = seq.next_element::<[String; LENGTH]>()? {
                self.pb.inc(1);
            }

            Ok(())
        }
    }

    deserializer.deserialize_seq(WordSquareVisitor { insert_record, records_to_read, pb })?;

    Ok(())
}

pub fn copy_json_to_database(
    json_file: String,
    word_square_db: &mut Database<Nu32, WordDbEntryCodec>,
    env: &Env,
    words: &Array2<char>,
    records_to_read: Option<u64>,
) -> Result<Vec<u32>, ErrT> {
    let mut wtxn = Some(env.write_txn()?);

    let mut changes = [0; 26];

    let reverse_word_map = create_reverse_word_map(words);

    read_json_with_visitor(json_file, records_to_read, &mut |data: Array2<char>,
                                            id,
                                            top_left_changed,
                                            top_left|
     -> Result<(), ErrT> {
        let word_db_entry = WordDbEntry {
            word_square: data.clone(),
            word_square_words: words_in_square(&data, &reverse_word_map).unwrap(),
        };
        let txn = wtxn.as_mut().expect("transaction should exist");

        word_square_db.put(txn, &id, &word_db_entry)?;

        if (id + 1) % TXN_SIZE == 0 {
            let txn = wtxn.take().expect("transaction should exist");
            txn.commit()?;
            wtxn = Some(env.write_txn()?);
        }

        if top_left_changed {
            changes[top_left as usize - 'a' as usize] = id;
        }

        Ok(())
    })?;

    wtxn.unwrap().commit()?;

    Ok(Vec::from(changes))
}

fn words_in_square(ws: &Array2<char>, reverse_word_map: &HashMap<Array1<char>, u16>) -> Result<HashSet<u16>, ErrT> {
    let mut result = HashSet::new();
    for i in 0..=1 {
        for word in ws.axis_iter(Axis(i)) {
            result.insert(*reverse_word_map.get(&word.to_owned()).expect("word not found"));
        }
    }

    assert!(result.len() == 2 * LENGTH);

    Ok(result)
}

/// Words are assumed sorted
pub fn read_words_from_file(file: String) -> Array2<char> {
    let mut raw_data = Vec::new();
    let mut count = 0;
    if let Ok(lines) = read_lines(file) {
        for line in lines.map_while(Result::ok) {
            let line = line.trim().to_ascii_lowercase();
            if line.chars().count() == LENGTH {
                raw_data.extend(line.chars());
                count += 1;
            }
        }
    }

    Array2::from_shape_vec((count, LENGTH), raw_data).unwrap()
}

fn create_reverse_word_map(words: &Array2<char>) -> HashMap<Array1<char>, u16> {
    let mut result = HashMap::new();
    for i in 0..words.shape()[0] {
        result.insert(words.slice(s![i, ..]).to_owned(), i as u16);
    }

    result
}

pub fn generate_prefixes_from_words(words: &Array2<char>) -> PrefixSet {
    let mut result: HashMap<Vec<char>, HashSet<u16>> = HashMap::new();
    for (word_id, word) in words.axis_iter(Axis(0)).enumerate() {
        for i in 0..LENGTH {
            let key_vec = word
                .slice(s![0..i])
                .to_owned()
                .into_iter()
                .collect::<Vec<_>>();
            let val = result.entry(key_vec).or_insert(HashSet::new());
            val.insert(word_id as u16);
        }
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
