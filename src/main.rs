use std::sync::Arc;

use clap::Parser;
// use ndarray::Array3;

mod args;
mod data;
mod solve;

use crate::data::{copy_json_to_database, read_words_from_file};
use crate::data::{ErrT, WordDb};
use crate::solve::solve_word_cubes;

fn main() -> Result<(), ErrT> {
    let cli = args::Args::parse();
    let db_folder = cli.db_folder;

    println!("Reading words from file");
    let words = read_words_from_file(cli.word_file);
    println!("Done");
    println!("Reading squares from json");
    copy_json_to_database(cli.json_file, db_folder.clone(), &words)?;
    println!("Done");

    let db = WordDb::new(db_folder)?;

    let result = solve_word_cubes(words, Arc::new(db))?;

    let file = std::fs::File::create(cli.out_file)?;
    let file = std::io::BufWriter::new(file);
    serde_json::to_writer(file, &result)?;

    Ok(())
}
