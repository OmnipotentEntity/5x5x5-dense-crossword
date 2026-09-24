use heed::Database;
use ndarray::Array2;

use crate::data::{generate_prefixes_from_words, Nu32, WordDbEntryCodec};

pub fn solve_word_cubes(words: Array2<char>, word_square_db: &Database<Nu32, WordDbEntryCodec>) {
    let prefix_set = generate_prefixes_from_words(&words);

    // word_square_db.iter();
}

pub fn solve_word_cubes_in_memory(words: Array2<char>, word_squares: Array3<char>) {
    let prefix_set = generate_prefixes_from_words(&words);

}
