use clap::Parser;

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

    /// Debugging test
    #[arg(short, long)]
    pub limit_ws: Option<u64>,
}
