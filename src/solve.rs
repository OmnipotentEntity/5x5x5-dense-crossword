use std::collections::HashSet;
use std::sync::Arc;
use std::thread::{self, available_parallelism};

use heed::{Database, Env, RoTxn};
use ndarray::{concatenate, s, stack, Array2, Array3, Array4, Axis};

use crate::data::{
    generate_prefixes_from_words, ErrT, Nu32, PrefixSet, WordDbEntry, WordDbEntryCodec, EMPTY_CELL,
    LENGTH,
};

/// This function generates all of the word cubes using the squares in word_square_db
/// word_square_db is assumed sorted by key, where the sorting function is only concerned with the
/// top left corner cell of the square.  This is because the sorting only exists to enforce the
/// fact that the word cube must be canonical.  This is an important optimization which reduces the
/// total amount of work by many times.
///
/// To understand canonical word cubes, consider a word square:
/// a b c d e
/// f g h i j
/// k l m n o
/// p q r s t
/// u v w x y
///
/// We can always generate another word square by exchanging rows with columns
/// a f k p u
/// b g l q v
/// e h m r w
/// d i n s x
/// e j o t y
///
/// We assert that a word square is canonical if the top row < top column lexiographically.  In the
/// top example we have if abcde < afkpu, the top square is canonical, while the bottom one is not,
/// and vice versa (here we are using these letters as variables, rather than concrete values.)
///
/// We can perform the same idea with word cubes, take this example 3x3x3 cube:
///
/// a b c | j k l | s t u
/// d e f | m n o | v w x
/// g h i | p q r | y z A
///
/// Now we have three axes on which to exchange, so in this case our three words, abc, adg, and
/// ajs.  Here there are a total of 6 ways to order these three words, and we define this word cube
/// as canonical if abc < adg < ajs.  This can be extended to any number of dimensions.
///
/// One can verify that exchanging any two axes results in a valid word cube:
///
/// (12):
/// a d g | j m p | s v y
/// b d h | k n q | t w z
/// c f i | l o r | u x A
///
/// (13):
/// a j s | b k t | c k t
/// d m v | e n w | f o x
/// g p y | h q z | i r A
///
/// (23):
/// a b c | d e f | g h i
/// j k l | m n o | p q r
/// s t u | v w x | y z A
///
/// Now we can consider the case during filling in a word cube:
///
/// a b c d e | A B C D E | _ _ _ _ _ | _ _ _ _ _ | _ _ _ _ _
/// f g h i j | F G H I J | _ _ _ _ _ | _ _ _ _ _ | _ _ _ _ _
/// k l m n o | K L M N O | _ _ _ _ _ | _ _ _ _ _ | _ _ _ _ _
/// p q r s t | P Q R S T | _ _ _ _ _ | _ _ _ _ _ | _ _ _ _ _
/// u v w x y | U V W X Y | _ _ _ _ _ | _ _ _ _ _ | _ _ _ _ _
///
/// We wish to place a word square in our third empty slice, so because of the requirement of
/// that the word square be canonical we only need to consider word squares with a top left entry
/// z, such that afk < aAz.  So we can simply start at the beginning if ak < aA, the index of k if
/// ak = aA, and if ak > aA, then our cube is already non-canonical by construction, so we can
/// disregard this situation entirely (we should never reach this case).

struct WordCubeData<'a> {
    word_cube: Array3<char>,
    word_square_db: Arc<Database<Nu32, WordDbEntryCodec>>,
    read_txn: &'a RoTxn<'a>,
    db_offsets: Vec<u32>,
    prefix_set: PrefixSet,
    used_words: HashSet<u16>,
}

pub fn solve_word_cubes(
    words: Array2<char>,
    word_square_db: Arc<Database<Nu32, WordDbEntryCodec>>,
    env: &Env,
    db_offsets: Vec<u32>,
) -> Result<Array4<char>, ErrT> {
    let prefix_set = generate_prefixes_from_words(&words);

    let num_threads = available_parallelism().unwrap().get() as u32;

    let mut threads = Vec::new();

    for i in 0..num_threads {
        let env = env.clone();
        let word_square_db = word_square_db.clone();
        let db_offsets = db_offsets.clone();
        let prefix_set = prefix_set.clone();
        let thread = thread::spawn(move || -> Result<Array4<char>, ErrT> {
            let mut results = Vec::new();
            let txn = env.read_txn().unwrap();
            let idx_mod = i as u32;
            let mut idx_mult = 0;
            while let Ok(Some(db_entry)) =
                word_square_db.get(&txn, &(idx_mult * num_threads + idx_mod))
            {
                let mut word_cube = Array3::from_elem((LENGTH, LENGTH, LENGTH), EMPTY_CELL);
                word_cube
                    .index_axis_mut(Axis(0), 0)
                    .assign(&db_entry.word_square.view());
                let mut wcd = WordCubeData {
                    word_cube: word_cube.clone(),
                    word_square_db: word_square_db.clone(),
                    read_txn: &txn,
                    db_offsets: db_offsets.clone(),
                    prefix_set: prefix_set.clone(),
                    used_words: db_entry.word_square_words,
                };
                solve_word_cube_impl(&mut wcd, 1, &mut results)?;

                idx_mult += 1;
            }


            let result = stack(
                Axis(0),
                &results.iter().map(|x| x.view()).collect::<Vec<_>>().as_slice()
            )?
            .to_owned();

            Ok(result)
        });

        threads.push(thread);
    }

    let mut result_vec = Vec::new();
    for thread in threads {
        let thread_results = thread.join().unwrap().unwrap();
        result_vec.push(thread_results);
    }
    let result = concatenate(
        Axis(0),
        &result_vec.iter().map(|x| x.view()).collect::<Vec<_>>().as_slice(),
    )?
    .to_owned();

    Ok(result)
}

fn solve_word_cube_impl(
    wcd: &mut WordCubeData,
    depth: usize,
    results: &mut Vec<Array3<char>>
) -> Result<(), ErrT> {
    let txn = wcd.read_txn;
    let mut idx = find_least_valid_index(wcd, depth);
    while let Some(wdb) = wcd.word_square_db.get(&txn, &idx)? {
        let mut handle_word_square_symmetry = |transposed: bool| -> Result<(), ErrT> {
            let did_place = try_place_word_square(wcd, &wdb, depth, transposed);
            if did_place {
                if depth == LENGTH - 1 {
                    println!("Found solution: {:?}", wcd.word_cube);
                    results.push(wcd.word_cube.clone());
                } else {
                    for &w in &wdb.word_square_words {
                        wcd.used_words.insert(w);
                    }
                    solve_word_cube_impl(wcd, depth + 1, results)?;
                    for &w in &wdb.word_square_words {
                        wcd.used_words.remove(&w);
                    }
                }
            }

            Ok(())
        };

        handle_word_square_symmetry(false)?;
        handle_word_square_symmetry(true)?;

        idx += 1;
    }

    Ok(())
}

fn find_least_valid_index(wcd: &WordCubeData, depth: usize) -> u32 {
    let col_prefix = String::from_iter(wcd.word_cube.slice(s![0, 0..depth, 0]));
    let depth_prefix = String::from_iter(wcd.word_cube.slice(s![0..depth, 0, 0]));

    if col_prefix < depth_prefix {
        return 0;
    }

    assert!(col_prefix == depth_prefix);

    wcd.db_offsets[wcd.word_cube[[0, depth, 0]] as usize - 'a' as usize]
}

fn try_place_word_square(
    wcd: &mut WordCubeData,
    wdb: &WordDbEntry,
    depth: usize,
    transpose: bool,
) -> bool {
    // If this word square contains a word we've already used
    if !wcd.used_words.is_disjoint(&wdb.word_square_words) {
        return false;
    }

    // Check each word along Axis(0) to check if it forms a valid prefix
    for i in 0..(LENGTH * LENGTH) {
        let major_idx = i / LENGTH;
        let minor_idx = i % LENGTH;
        let prefix = Vec::from_iter(
            wcd.word_cube
            .slice(s![0..depth, major_idx, minor_idx])
            .iter()
            .copied()
            .chain(std::iter::once(if !transpose {
                wdb.word_square[[major_idx, minor_idx]]
            } else {
                wdb.word_square[[minor_idx, major_idx]]
            }))
        );

        match wcd.prefix_set.get(&prefix) {
            Some(prefixes) if prefixes.is_subset(&wcd.used_words) => return false,
            None => return false,
            _ => (),
        }
    }

    // If we have no conflicts
    // actually write into the word cube and signal that we did
    wcd.word_cube
        .index_axis_mut(Axis(0), depth)
        .assign(&wdb.word_square.view());

    true
}
