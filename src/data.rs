use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, prelude::*, BufRead};
use std::mem::size_of;
use std::path::Path;

use heed::{
    byteorder::NativeEndian, types::U32, BoxedError, BytesDecode, BytesEncode, Database, Env,
};
use indicatif::ProgressBar;
use ndarray::{s, Array, Array2, ArrayView1, /*Array3, ArrayView,*/ Axis};

pub const EMPTY_CELL: char = ' ';
pub const LENGTH: usize = 5;
const TXN_SIZE: u32 = 1024 * 1024;

pub type PrefixSet = HashMap<Vec<char>, HashSet<u16>>;
pub type ErrT = Box<dyn Error + Send + Sync>;

#[derive(Debug)]
pub struct JsonParseError {
    cause: String,
}

impl JsonParseError {
    fn new(cause: String) -> JsonParseError {
        JsonParseError {
            cause: String::from(cause),
        }
    }
}

impl fmt::Display for JsonParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unable to read json file: {:?}", self.cause)
    }
}

impl Error for JsonParseError {}

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

/// hahaha, this is way too few states, and the wrong states, but that's OK, because this isn't a
/// real parser.
#[derive(PartialEq, Eq, Debug)]
enum JsonReaderState {
    OuterArray,
    InnerArray,
    String,
}

// The structure of the json file is known, so a Q&D parser is used here
fn read_json_with_visitor<T: FnMut(Array2<char>, u32, bool) -> Result<(), ErrT>>(
    json_file: String,
    insert_record: &mut T,
) -> Result<(), ErrT> {
    let file = File::open(json_file)?;
    let file_size = file.metadata().unwrap().len();
    // There are N quoted strings, each separated by commas, that's (N + 3) * N - 1
    // Finally, each record has an opening brace, closing brace, and trailing comma
    let record_length = (LENGTH + 3) * LENGTH + 2;
    // The entire is an opening bracket, followed by any number of records, followed by a closing
    // bracket
    // However, the final record does not have a trailing comma; therefore, we have overcounted by
    // one, so this is actually file_size - 2 + 1
    let records = (file_size - 1) / (record_length as u64);
    let buf = io::BufReader::new(file);

    let mut state_vec: Vec<JsonReaderState> = Vec::new();
    let mut data_vec: Array2<char> = Array2::from_elem((LENGTH, LENGTH), EMPTY_CELL);
    let mut data_idx_major: usize = 0;
    let mut data_idx_minor: usize = 0;
    let mut db_id: u32 = 0;

    let pb = ProgressBar::new(records);

    let mut last_top_left = ' ';
    let mut top_left_changed = false;

    for byte in buf.bytes() {
        if byte.is_err() {
            break;
        }

        let ascii = byte? as char;

        match ascii {
            '[' => match state_vec.last() {
                None => state_vec.push(JsonReaderState::OuterArray),
                Some(JsonReaderState::OuterArray) => state_vec.push(JsonReaderState::InnerArray),
                _ => {
                    return Err(Box::new(JsonParseError::new(format!(
                        "Wrong state when reading open bracket: {:?}",
                        state_vec.last()
                    ))));
                }
            },

            ',' => match state_vec.last() {
                Some(JsonReaderState::OuterArray) => {
                    data_idx_major = 0;
                    data_idx_minor = 0;
                }
                Some(JsonReaderState::InnerArray) => {
                    data_idx_minor = 0;
                }
                _ => {
                    return Err(Box::new(JsonParseError::new(format!(
                        "Wrong state when reading comma: {:?}",
                        state_vec.last()
                    ))));
                }
            },

            ']' => match state_vec.last() {
                Some(JsonReaderState::OuterArray) => {
                    break;
                }
                Some(JsonReaderState::InnerArray) => {
                    if data_idx_major != LENGTH {
                        return Err(Box::new(JsonParseError::new(format!(
                            "Wrong number of elements in inner array: expected {}, got {}",
                            LENGTH, data_idx_major
                        ))));
                    };

                    insert_record(data_vec.clone(), db_id, top_left_changed)?;
                    db_id += 1;
                    pb.inc(1);
                    state_vec.pop();
                }
                _ => {
                    return Err(Box::new(JsonParseError::new(format!(
                        "Wrong state when reading close bracket: {:?}",
                        state_vec.last()
                    ))));
                }
            },

            '"' => match state_vec.last() {
                Some(JsonReaderState::InnerArray) => {
                    state_vec.push(JsonReaderState::String);
                }
                Some(JsonReaderState::String) => {
                    if data_idx_minor != LENGTH {
                        return Err(Box::new(JsonParseError::new(format!(
                            "Wrong number of elements in string: expected {}, got {}",
                            LENGTH, data_idx_minor
                        ))));
                    }
                    data_idx_major += 1;
                    state_vec.pop();
                }
                _ => {
                    return Err(Box::new(JsonParseError::new(format!(
                        "Wrong state when reading quote: {:?}",
                        state_vec.last()
                    ))));
                }
            },

            x @ 'a'..='z' => {
                if state_vec.last() == Some(&JsonReaderState::String) {
                    // Ensure that the database is sorted by the top left value.
                    if (data_idx_major, data_idx_minor) == (0, 0) {
                        if x < last_top_left {
                            return Err(Box::new(JsonParseError::new(format!(
                                "Json is not sorted at index {}: {} < {}",
                                db_id, x, last_top_left
                            ))));
                        } else if x != last_top_left {
                            last_top_left = x;
                            top_left_changed = true;
                        } else {
                            top_left_changed = false;
                        }
                    }
                    data_vec[(data_idx_major, data_idx_minor)] = x;
                    data_idx_minor += 1;
                } else {
                    return Err(Box::new(JsonParseError::new(format!(
                        "Wrong state when reading characters: {:?}",
                        state_vec.last()
                    ))));
                }
            }

            x if x.is_whitespace() => {}

            x => {
                return Err(Box::new(JsonParseError::new(format!(
                    "Unexpected character: '{}'",
                    x
                ))));
            }
        };
    }

    Ok(())
}

pub fn copy_json_to_database(
    json_file: String,
    word_square_db: &mut Database<Nu32, WordDbEntryCodec>,
    env: &Env,
    words: &Array2<char>,
) -> Result<Vec<u32>, ErrT> {
    let mut wtxn = Some(env.write_txn()?);

    let mut changes = Vec::new();

    read_json_with_visitor(json_file, &mut |data: Array2<char>,
                                            id,
                                            top_left_changed|
     -> Result<(), ErrT> {
        let word_db_entry = WordDbEntry {
            word_square: data.clone(),
            word_square_words: words_in_square(&data, words).unwrap(),
        };
        let mut txn = wtxn.take().expect("transaction should exist");

        word_square_db.put(&mut txn, &id, &word_db_entry)?;

        if (id + 1) % TXN_SIZE == 0 {
            txn.commit()?;
            wtxn = Some(env.write_txn()?);
        }
        if top_left_changed {
            changes.push(id);
        }

        Ok(())
    })?;

    wtxn.unwrap().commit()?;

    Ok(changes)
}

fn words_in_square(ws: &Array2<char>, words: &Array2<char>) -> Result<HashSet<u16>, ErrT> {
    let mut result = HashSet::new();
    for i in 0..=1 {
        for word in ws.axis_iter(Axis(i)) {
            result.insert(binary_search_words(word, words)?);
        }
    }

    assert!(result.len() == 2 * LENGTH);

    Ok(result)
}

pub fn binary_search_words(word: ArrayView1<char>, words: &Array2<char>) -> Result<u16, ErrT> {
    // binary search
    let mut low = 0;
    let mut high = words.shape()[0];

    while low <= high {
        let mid = (low + high) / 2;
        if words.slice(s![mid, ..]) == word {
            return Ok(mid as u16);
        } else if word.iter().collect::<String>()
            < words.slice(s![mid, ..]).iter().collect::<String>()
        {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    // could not find word in list
    return Err(Box::new(JsonParseError::new(
        "Unable to find word in list".into(),
    )));
}

/// Words are assumed sorted
pub fn read_words_from_file(file: String) -> Array2<char> {
    let mut result = Array2::<char>::default((0, LENGTH));
    if let Ok(lines) = read_lines(file) {
        for line in lines.map_while(Result::ok) {
            if line.len() == LENGTH {
                result
                    .append(
                        Axis(0),
                        Array::from_iter(line.to_ascii_lowercase().chars())
                            .into_shape_with_order((1, LENGTH))
                            .expect("error with word length")
                            .view(),
                    )
                    .expect("error with word length");
            }
        }
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
