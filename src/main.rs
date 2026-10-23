use std::sync::Arc;
use std::time::Instant;

mod args;
mod data;
mod solve;

use crate::args::ARGS;
use crate::data::{copy_json_to_database, read_words_from_file};
use crate::data::{ErrT, WordDb};
use crate::solve::solve_word_cubes;

fn main() -> Result<(), ErrT> {
    println!("Reading words from file");
    let words = read_words_from_file(ARGS.word_file.clone());
    println!("Done");
    println!("Reading squares from json");
    copy_json_to_database(&words)?;
    println!("Done");

    let db = WordDb::new()?;

    if !ARGS.benchmark {
        let result = solve_word_cubes(words, Arc::new(db))?;
        let file = std::fs::File::create(ARGS.out_file.clone())?;
        let file = std::io::BufWriter::new(file);
        serde_json::to_writer(file, &result)?;
    } else {
        println!("Benchmark Mode Active!");
        let now = Instant::now();
        let _ = solve_word_cubes(words, Arc::new(db));
        let elapsed_time = now.elapsed();
        println!(
            "Running 1 billion iterations took {} seconds.",
            elapsed_time.as_micros() as f64 / 1e6
        );
    };

    Ok(())
}
