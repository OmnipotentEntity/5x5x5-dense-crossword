use std::sync::LazyLock;

use clap::Parser;

pub static ARGS: LazyLock<Args> = LazyLock::new(|| Args::parse());

/// Command line arguments for the program
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Name of the file containing the 5x5 dense crosswords
    #[arg(short, long, default_value_t = String::from("scrabble.json"))]
    pub json_file: String,

    #[arg(short, long, default_value_t = String::from("CSW24.txt"))]
    pub word_file: String,

    /// Output file
    #[arg(short, long, default_value_t = String::from("dense5x5x5.json"))]
    pub out_file: String,

    /// Database
    #[arg(short, long, default_value_t = String::from("ws.db"))]
    pub db_folder: String,

    /// Benchmark mode
    #[arg(short, long, default_value_t = false)]
    pub benchmark: bool,
}
