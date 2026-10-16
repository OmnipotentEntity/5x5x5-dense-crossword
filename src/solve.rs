use std::cmp::Ordering;
use std::sync::Arc;
use std::thread::{self, available_parallelism};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::{concatenate, s, stack, Array2, Array3, Array4, ArrayView1, Axis};

use crate::data::{
    generate_prefixes_from_words, ErrT, PrefixSet, WordDb, WordDbEntry, EMPTY_CELL, LENGTH, LETTERS,
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

struct WordCubeData {
    word_cube: Array3<u8>,
    word_square_db: Arc<WordDb>,
    prefix_set: PrefixSet,
    used_words: Vec<u16>,
}

pub fn solve_word_cubes(
    words: Array2<u8>,
    word_square_db: Arc<WordDb>,
) -> Result<Array4<u8>, ErrT> {
    let prefix_set = generate_prefixes_from_words(&words);

    let num_threads = available_parallelism().unwrap().get();

    let mp = Arc::new(MultiProgress::new());
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("#>-");

    let mut threads = Vec::new();

    for i in 0..num_threads {
        let thread_total = (word_square_db.count + num_threads - 1 - i) / num_threads;
        let mp_thread = mp.clone();
        let pb = mp_thread.add(ProgressBar::new(thread_total as u64));
        pb.set_style(style.clone());
        pb.set_message(format!("Thread {}", i));

        let word_square_db = word_square_db.clone();
        let prefix_set = prefix_set.clone();
        let thread = thread::spawn(move || -> Result<Array4<u8>, ErrT> {
            let mut results = Vec::new();
            let mut idx_mult = 0;
            while let Some(db_entry) = word_square_db.get(idx_mult * num_threads + i) {
                let mut word_cube = Array3::from_elem((LENGTH, LENGTH, LENGTH), EMPTY_CELL);
                word_cube
                    .index_axis_mut(Axis(0), 0)
                    .assign(&db_entry.word_square_view());
                let mut wcd = WordCubeData {
                    word_cube: word_cube.clone(),
                    word_square_db: word_square_db.clone(),
                    prefix_set: prefix_set.clone(),
                    used_words: Vec::from(db_entry.word_square_words),
                };
                solve_word_cube_impl(&mut wcd, 1, &mut results, &mp_thread)?;

                idx_mult += 1;
                pb.inc(1);
            }

            pb.finish_with_message(format!("Thread {} complete", i));

            let result = if results.is_empty() {
                Array4::from_elem((0, LENGTH, LENGTH, LENGTH), EMPTY_CELL)
            } else {
                stack(
                    Axis(0),
                    &results.iter().map(|x| x.view()).collect::<Vec<_>>(),
                )?
            };

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
        &result_vec
            .iter()
            .map(|x| x.view())
            .collect::<Vec<_>>()
            .as_slice(),
    )?
    .to_owned();

    Ok(result)
}

#[derive(Clone, PartialEq)]
enum PrefixMemo {
    NotInit,
    False,
    True,
}

fn solve_word_cube_impl(
    wcd: &mut WordCubeData,
    depth: usize,
    results: &mut Vec<Array3<u8>>,
    mp: &MultiProgress,
) -> Result<(), ErrT> {
    // The most expensive thing we do is check the prefix array.  This is a memo for just the
    // prefixes we care about for this wordsquare
    let mut prefix_memo = Array3::from_elem((LENGTH, LENGTH, LETTERS), PrefixMemo::NotInit);
    let mut idx = find_least_valid_index(wcd, depth);
    let word_square_db = wcd.word_square_db.clone();
    while let Some(wdb) = word_square_db.get(idx) {
        let mut handle_word_square_symmetry = |transposed: bool| -> Result<(), ErrT> {
            let did_place = try_place_word_square(wcd, &wdb, depth, transposed, &mut prefix_memo);
            if did_place {
                if depth == LENGTH - 1 {
                    let _ = mp.println(format!("Found solution: {:?}", wcd.word_cube));
                    results.push(wcd.word_cube.clone());
                } else {
                    set_insert(&mut wcd.used_words, wdb.word_square_words);
                    solve_word_cube_impl(wcd, depth + 1, results, mp)?;
                    set_difference(&mut wcd.used_words, wdb.word_square_words);
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

fn find_least_valid_index(wcd: &WordCubeData, depth: usize) -> usize {
    let col_prefix = wcd.word_cube.slice(s![0, 0..depth, 0]);
    let depth_prefix = wcd.word_cube.slice(s![0..depth, 0, 0]);

    if array_view_lt(col_prefix, depth_prefix) {
        return 0;
    }

    assert!(col_prefix == depth_prefix);

    wcd.word_square_db.changes[wcd.word_cube[[0, depth, 0]] as usize - 'a' as usize]
}

fn try_place_word_square(
    wcd: &mut WordCubeData,
    wdb: &WordDbEntry,
    depth: usize,
    transpose: bool,
    prefix_memo: &mut Array3<PrefixMemo>,
) -> bool {
    // If this word square contains a word we've already used
    // if !wcd.used_words.is_disjoint(&wdb.word_square_words) {
    if !is_disjoint(wcd.used_words.as_slice(), wdb.word_square_words) {
        return false;
    }

    let wsv = wdb.word_square_view();

    // Check each word along Axis(0) to check if it forms a valid prefix
    let mut prefix = [0u8; LENGTH];
    let slice_len = depth + 1;

    for major_idx in 0..LENGTH {
        for minor_idx in 0..LENGTH {
            let ws_letter = if !transpose {
                wsv[[major_idx, minor_idx]]
            } else {
                wsv[[minor_idx, major_idx]]
            };

            match prefix_memo[[major_idx, minor_idx, ws_letter as usize - 'a' as usize]] {
                PrefixMemo::False => return false,
                PrefixMemo::True => continue,
                _ => (),
            }
            for d in 0..depth {
                prefix[d] = wcd.word_cube[[d, major_idx, minor_idx]];
            }
            prefix[depth] = ws_letter;

            match wcd.prefix_set.get(&prefix[..slice_len]) {
                Some(prefixes) if is_subset(&wcd.used_words, prefixes) => {
                    prefix_memo[[major_idx, minor_idx, ws_letter as usize - 'a' as usize]] =
                        PrefixMemo::False;
                    return false;
                }
                None => {
                    prefix_memo[[major_idx, minor_idx, ws_letter as usize - 'a' as usize]] =
                        PrefixMemo::False;
                    return false;
                }
                _ => {
                    prefix_memo[[major_idx, minor_idx, ws_letter as usize - 'a' as usize]] =
                        PrefixMemo::True;
                }
            }
        }
    }

    // If we have no conflicts
    // actually write into the word cube and signal that we did
    let wsv = wdb.word_square_view();
    let mut target = wcd.word_cube.index_axis_mut(Axis(0), depth);
    if !transpose {
        target.assign(&wsv)
    } else {
        target.assign(&wsv.reversed_axes())
    }

    true
}

fn set_insert<T: Ord + Copy>(dest: &mut Vec<T>, src: &[T; 2 * LENGTH]) {
    dest.extend_from_slice(src);
    dest.sort_unstable();
}

/// This is not a general set difference function!  This assumes every element in b is also in a.
fn set_difference<T: Ord>(a: &mut Vec<T>, b: &[T; 2 * LENGTH]) {
    let mut j = 0;
    a.retain(|x| {
        // Advance `j` in `b` as long as `b[j]` is smaller than `x`
        while j < b.len() && &b[j] < x {
            j += 1;
        }
        // If we found `x` in `b`, filter it out (return false)
        if j < b.len() && &b[j] == x {
            j += 1; // Move past this matched element
            false
        } else {
            true // Keep element in `a`
        }
    });
}

/// Checks if b is a subset of a
fn is_subset<T: Ord>(a: &Vec<T>, b: &Vec<T>) -> bool {
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        let a_val = &a[i];
        let b_val = &b[j];

        match a_val.cmp(b_val) {
            Ordering::Equal => {
                i += 1;
                j += 1
            }
            Ordering::Less => i += 1,
            Ordering::Greater => return false,
        }
    }

    j == b.len()
}

fn is_disjoint<T: Ord>(a: &[T], b: &[T]) -> bool {
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        let a_val = &a[i];
        let b_val = &b[j];

        match a_val.cmp(b_val) {
            Ordering::Equal => return false,
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }

    return true;
}

/// Assumes a and b are the same length
fn array_view_lt(a: ArrayView1<u8>, b: ArrayView1<u8>) -> bool {
    let mut i = 0;
    let len = a.shape()[0];

    while i < len {
        let a_val = &a[i];
        let b_val = &b[i];

        match a_val.cmp(b_val) {
            Ordering::Equal => (),
            Ordering::Less => return true,
            Ordering::Greater => return false,
        }

        i += 1;
    }

    false
}
