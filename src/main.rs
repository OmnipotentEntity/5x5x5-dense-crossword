use std::error::Error;

use clap::Parser;
// use heed::{Database, EnvOpenOptions};
use ndarray::Array3;
use tempfile;

mod args;
mod data;
mod solve;

use crate::data::{copy_json_to_database, read_json_to_memory, read_words_from_file};
use crate::data::{ErrT, Nu32, WordDbEntryCodec};
use crate::solve::solve_word_cubes;

fn main() -> Result<(), ErrT> {
    let cli = args::Args::parse();
    let path = tempfile::tempdir().expect("Could not create tempdir");

    // let env = unsafe {
    //     EnvOpenOptions::new()
    //         .map_size(100 * 1024 * 1024 * 1024) // 100 GiB
    //         .max_dbs(1)
    //         .open(path)?
    // };

    // let mut wtxn = env.write_txn()?;

    // let mut word_square_db: Database<Nu32, WordDbEntryCodec> =
    //     env.create_database(&mut wtxn, None)?;

    // wtxn.commit()?;

    println!("Reading words from file");
    let words = read_words_from_file(cli.word_file);
    println!("Done");
    println!("Reading squares from json");
    //copy_json_to_database(cli.json_file, &mut word_square_db, &env, &words)?;
    let word_squares = read_json_to_memory(cli.json_file)?;
    // let word_square_db: Array3<char> = read_json_to_array(cli.json_file)?;
    println!("Done");

    // solve_word_cubes(words, &word_square_db);
    solve_word_cubes_in_memory(words, word_squares);

    Ok(())
}
