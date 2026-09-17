use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::iter;
use std::path::Path;

use indicatif::{MultiProgress, ProgressBar};
use ndarray::{s, Array, Array2, Axis, IxDyn, SliceInfo, SliceInfoElem};

const EMPTY_CELL: char = ' ';
const DEFAULT_DIMS: usize = 3;
const DEFAULT_LENGTH: usize = 5;

type PrefixSet = HashMap<Vec<char>, HashSet<Vec<char>>>;

fn read_file(file: String, length: usize) -> Array2<char> {
    let mut result = Array2::<char>::default((0, length));
    if let Ok(lines) = read_lines(file) {
        for line in lines.map_while(Result::ok) {
            if line.len() == length {
                result
                    .append(
                        Axis(0),
                        Array::from_iter(line.to_ascii_lowercase().chars())
                            .into_shape_with_order((1, length))
                            .expect("error with word length")
                            .view(),
                    )
                    .expect("error with word length");
            }
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

fn build_prefix_set(words: Array2<char>, length: usize) -> PrefixSet {
    let mut result = HashMap::new();
    for i in 0..=length {
        for row in words.rows().into_iter() {
            let key_vec = row
                .slice(s![0..i])
                .to_owned()
                .into_iter()
                .collect::<Vec<_>>();
            let val = result.entry(key_vec).or_insert(HashSet::new());
            val.insert(row.to_owned().into_iter().collect());
        }
    }

    result
}

struct SolverState {
    // These are all of the valid words
    words: HashSet<Vec<char>>,

    // These are the pared down sets of words for each column, sorted by score
    words_per_column: Array<Option<Vec<Vec<char>>>, IxDyn>,

    // These are all of the words used so far
    used_words: HashSet<Vec<char>>,

    // This field is for backtracking, for each entry in the working dimension, we try to place a
    // word, and if we fail then we jump back to a previous word.  These values are indices into
    // the array representing our progress
    iters: Array<Option<usize>, IxDyn>,

    // These are words that we've previously calculated to conflict along a given axis, and this
    // axis hasn't changed since the last time we visited this column
    conflicts: Array<Option<HashSet<Vec<char>>>, IxDyn>,

    // The working output of the solver.  A value of EMPTY_CELL represents unset.
    puzzle: Array<char, IxDyn>,
}

impl SolverState {
    fn init(words: Array2<char>, length: usize, dims: usize) -> SolverState {
        SolverState {
            words: HashSet::from_iter(words.axis_iter(Axis(0)).map(|x| x.to_vec())),

            words_per_column: Array::from_iter(iter::repeat_n(None, length.pow(dims as u32 - 1)))
                .into_shape_with_order(IxDyn(
                    Vec::from_iter(iter::repeat_n(length, dims - 1)).as_slice(),
                ))
                .unwrap(),
            used_words: HashSet::new(),
            iters: Array::from_iter(iter::repeat_n(None, length.pow(dims as u32 - 1)))
                .into_shape_with_order(IxDyn(
                    Vec::from_iter(iter::repeat_n(length, dims - 1)).as_slice(),
                ))
                .unwrap(),
            conflicts: Array::from_iter(iter::repeat_n(
                None,
                (dims - 1) * length.pow(dims as u32 - 1),
            ))
            .into_shape_with_order(IxDyn(
                &[
                    Vec::from_iter(iter::repeat_n(length, dims - 1)).as_slice(),
                    &[dims - 1],
                ]
                .concat(),
            ))
            .unwrap(),
            puzzle: Array::from_shape_fn(
                Vec::from_iter(iter::repeat_n(length, dims)).as_slice(),
                |_| EMPTY_CELL,
            ),
        }
    }
}

fn index_to_column(idx: usize, length: usize, dims: usize) -> Vec<usize> {
    let mut res: Vec<usize> = iter::repeat_n(0, dims - 1).collect();
    let mut idx = idx;
    let mut res_idx = 0;

    while idx != 0 {
        res[res_idx] = idx % length;
        idx /= length;
        res_idx += 1;
    }

    res
}

fn column_to_index(vec_idx: Vec<usize>, length: usize) -> usize {
    let mut base = 1;
    let mut result = 0;

    for idx in vec_idx.iter() {
        result += base * idx;
        base *= length;
    }

    result
}

fn slice_info_builder(
    i: usize,
    j: usize,
    vec_idx: Vec<usize>,
) -> SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
    let mut result = Vec::new();

    for (idx, dim_idx) in vec_idx.iter().enumerate() {
        if idx == i {
            result.push(SliceInfoElem::Slice {
                start: 0,
                end: Some(*dim_idx as isize),
                step: 1,
            });
        } else {
            result.push(SliceInfoElem::Index(*dim_idx as isize));
        }
    }

    result.push(SliceInfoElem::Index(j as isize));

    SliceInfo::try_from(result).unwrap()
}

fn column_slice_info_builder(
    vec_idx: Vec<usize>,
    length: isize,
) -> SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
    let mut slice_info = vec_idx
        .iter()
        .map(|x| SliceInfoElem::Index(*x as isize))
        .collect::<Vec<SliceInfoElem>>();
    slice_info.push(SliceInfoElem::Slice {
        start: 0,
        end: Some(length as isize),
        step: 1,
    });

    SliceInfo::try_from(slice_info).unwrap()
}

fn main() {
    let mut args = env::args();
    let _ = args.next();
    let file_name: String = args.next().unwrap_or(String::from("CSW24.txt"));
    let dims: usize = args
        .next()
        .and_then(|x| x.parse().ok())
        .unwrap_or(DEFAULT_DIMS);
    let length: usize = args
        .next()
        .and_then(|x| x.parse().ok())
        .unwrap_or(DEFAULT_LENGTH);

    let valid_words = read_file(file_name, length);
    let prefixes = build_prefix_set(valid_words.clone(), length);

    let mut ss = SolverState::init(valid_words, length.into(), dims.into());

    let mut idx = 0;
    let m = MultiProgress::new();
    let idx_pb = ProgressBar::new((length as u64).pow(dims as u32 - 1));
    idx_pb.update(|x| x.set_pos(idx as u64));

    let dim_pb = ProgressBar::new(dims as u64 - 1);
    let word_pb = ProgressBar::new(ss.words.len() as u64);
    let iter_pbs = (0..(length as usize).pow(dims as u32 - 1))
        .map(|_| ProgressBar::new(ss.words.len() as u64))
        .collect::<Vec<_>>();

    m.add(idx_pb.clone());
    m.add(dim_pb.clone());
    m.add(word_pb.clone());
    for iter_pb in iter_pbs.iter() {
        m.add(iter_pb.clone());
    }

    'outer: loop {
        //println!("loop top, idx: {}", idx);
        //println!("used words len: {}", ss.used_words.len());
        let vec_idx = index_to_column(idx, length.into(), dims.into());

        // TODO: Need to restructure to the following:
        // For each axis
        // 1. If our conflicts for this axis is not yet initialized (is None) then
        //   a. Examine every word along this axis.
        //   b. If the word conflicts along this axis add it to the conflicts
        // 2.
        //   a. Pare down the set of open words and move to the next axis
        // 3. If all of the words are eliminated at any point, backtrack along that axis
        // 4. If we can place a word, then place the first word remaining along the axis
        //    (saving this as the working iterator for this column)
        //
        // This places a lot of work upfront, some of which is thrown away during backtracking,
        // but it should make for a lot less redundant work performed.

        // If we have an initialized words_per_column, then use that.
        if ss.words_per_column[vec_idx.as_slice()].is_none() {
            //println!("Reinit words per column");
            // Our iter for this should also be none
            assert!(ss.iters[vec_idx.as_slice()].is_none());
            // And at least one of the conflicts should be none

            let mut leftover_words: HashSet<_> =
                ss.words.difference(&ss.used_words).cloned().collect();
            //println!("leftover_words len: {}", leftover_words.len());
            for i in 0..dims as usize - 1 {
                dim_pb.update(|x| x.set_pos(i as u64));
                // Need to check most significant first
                let i = dims as usize - 2 - i;
                //println!("Dim: {}", i);

                let conflict_idx = [vec_idx.clone(), vec![i]].concat();
                if ss.conflicts[conflict_idx.as_slice()].is_some() {
                    //println!("Skipping filtering for dim");
                    // If we already calculated the conflicts for this axis, skip
                    leftover_words = leftover_words
                        .difference(&ss.conflicts[conflict_idx.as_slice()].as_ref().unwrap())
                        .cloned()
                        .collect();
                    continue;
                }

                //println!("Beyond continue");
                ss.conflicts[conflict_idx.as_slice()] = Some(HashSet::new());

                word_pb.update(|x| x.set_len(leftover_words.len() as u64));

                for (word_idx, word) in leftover_words.iter().enumerate() {
                    word_pb.update(|x| x.set_pos(word_idx as u64));

                    for j in 0..length as usize {
                        let mut puzzle_prefix_for_dim_i = Vec::from_iter(
                            ss.puzzle
                                .slice(slice_info_builder(i, j, vec_idx.clone()))
                                .to_owned()
                                .into_iter(),
                        );
                        puzzle_prefix_for_dim_i.push(word[j]);

                        if let Some(prefixes) = prefixes.get(&puzzle_prefix_for_dim_i) {
                            let valid_prefixes = prefixes - &ss.used_words;
                            if valid_prefixes.len() == 0 {
                                ss.conflicts[conflict_idx.as_slice()]
                                    .as_mut()
                                    .unwrap()
                                    .insert(word.clone());
                            }
                        } else {
                            ss.conflicts[conflict_idx.as_slice()]
                                .as_mut()
                                .unwrap()
                                .insert(word.clone());
                        }
                    }
                }

                // Remove conflicts from consideration on the next axis
                leftover_words = leftover_words
                    .difference(&ss.conflicts[conflict_idx.as_slice()].as_ref().unwrap())
                    .cloned()
                    .collect();

                //println!("Leftover words len: {}", leftover_words.len());

                // if we've run out of words then rollback along the dimension we are considering
                if leftover_words.len() == 0 {
                    //println!("Rolling back along dim {}", i);
                    let mut rollback_vec_idx = vec_idx;
                    rollback_vec_idx[i] -= 1;

                    let rollback_idx = column_to_index(rollback_vec_idx, length);
                    //println!("current index {}, target index {}", idx, rollback_idx);

                    for loop_idx in rollback_idx..=idx {
                        let loop_vec_idx = index_to_column(loop_idx, length, dims);
                        if loop_idx != idx {
                            let puz_word = &ss
                                .puzzle
                                .slice(column_slice_info_builder(
                                    loop_vec_idx.clone(),
                                    length as isize,
                                ))
                                .to_owned()
                                .into_iter()
                                .collect::<Vec<char>>();
                            //println!("Removing {} from used_words", puz_word.into_iter().collect::<String>());
                            //println!("before: {}", ss.used_words.len());
                            ss.used_words.remove(puz_word);
                            //println!("after: {}", ss.used_words.len());
                        }

                        if loop_idx != rollback_idx {
                            // Don't clear the iter of our destination
                            ss.words_per_column[loop_vec_idx.as_slice()] = None;
                            ss.iters[loop_vec_idx.as_slice()] = None;
                            //println!("Rolling back conflicts, from 0 to {}", i);
                            for j in 0..=i {
                                // We could probably do some neat math with mods and so on, but instead
                                // I'll do something a bit easier to conceptualize.  I'll just probe
                                // each direction and see if it's in the range [idx, rollback_idx], and
                                // if so, rollback that conflict set

                                let mut probe_vec_idx = loop_vec_idx.clone();
                                if probe_vec_idx[j] != 0 {
                                    probe_vec_idx[j] -= 1;
                                    let probe_idx = column_to_index(probe_vec_idx, length);
                                    if probe_idx >= rollback_idx {
                                        // println!(
                                        //     "Resetting conflicts for idx {} along dim {}",
                                        //     loop_idx, j
                                        // );
                                        // will always be less than idx
                                        ss.conflicts
                                            [[loop_vec_idx.clone(), vec![j]].concat().as_slice()] =
                                            None;
                                    }
                                }
                            }
                        }
                    }

                    idx = rollback_idx;
                    idx_pb.update(|x| x.set_pos(idx as u64));

                    assert!(ss.used_words.len() == idx);

                    // println!("Continue outer");
                    continue 'outer;
                }
            }

            // At this point, leftover_words is exactly our candidate words for this location.
            // Now we need to make a sorted vector that contains these words.  The sorting order is
            // vibes, (i.e. the words least likely to cause conflicts on any of the subsequent axes)

            // println!("leftover words len: {}", leftover_words.len());
            let mut candidate_words = leftover_words
                .clone()
                .into_iter()
                .collect::<Vec<Vec<char>>>();
            // println!("Sort candidate words");
            candidate_words.sort_by_cached_key(|x| {
                let mut scores_along_dim = iter::repeat_n(1, dims - 1).collect::<Vec<usize>>();
                for i in 0..dims - 1 as usize {
                    let mut examine_idx = vec_idx.clone();
                    examine_idx[i] += 1;
                    // We're off the grid
                    if examine_idx[i] >= length {
                        // println!("early out dim {}", i);
                        scores_along_dim[i] = 1;
                        continue;
                    }

                    let mut score = ss.words.len();
                    for j in 0..length as usize {
                        let mut puzzle_prefix_for_dim_i = ss
                            .puzzle
                            .slice(slice_info_builder(i, j, examine_idx.clone()))
                            .to_owned()
                            .into_iter()
                            .collect::<Vec<char>>();
                        puzzle_prefix_for_dim_i[vec_idx[i]] = x[j];

                        if let Some(letter_prefixes) = prefixes.get(&puzzle_prefix_for_dim_i) {
                            score = score.min(letter_prefixes.len());
                        } else {
                            score = 0;
                        }
                    }

                    scores_along_dim[i] = score;
                }

                let result = scores_along_dim
                    .into_iter()
                    .fold(ss.words.len(), |acc, x| acc.min(x));
                result
            });

            candidate_words.reverse();
            // println!(
            //     "End sort of candidate words, best word: {}",
            //     candidate_words[0].iter().cloned().collect::<String>()
            // );

            let word_count = candidate_words.len() as u64;
            ss.words_per_column[vec_idx.as_slice()] = Some(candidate_words);
            ss.iters[vec_idx.as_slice()] = Some(0);
            iter_pbs[idx].update(|x| {
                x.set_len(word_count);
                x.set_pos(0)
            });
        }

        // Now that candidate words is fully initialized, use the saved index, and
        // place the next word.

        let iter_idx = ss.iters[vec_idx.as_slice()].as_mut().unwrap();
        let word_len = ss.words_per_column[vec_idx.as_slice()]
            .as_ref()
            .unwrap()
            .len();
        // println!("iter_idx: {}, word_per_column.len() {}", iter_idx, word_len);
        if *iter_idx >= word_len {
            // We've run out of words, backtrack
            // If we have exhausted all possible words, then we error and exit
            if idx == 0 {
                println!("Failure");
                return;
            }
            // println!("Clear words (bottom)");
            ss.words_per_column[vec_idx.as_slice()] = None;
            ss.iters[vec_idx.as_slice()] = None;
            ss.conflicts[[vec_idx.clone(), vec![dims - 2]].concat().as_slice()] = None;
            idx -= 1;
            let vec_idx = index_to_column(idx, length, dims);
            ss.used_words.remove(
                &ss.puzzle
                    .slice(column_slice_info_builder(vec_idx, length as isize))
                    .to_owned()
                    .into_iter()
                    .collect::<Vec<char>>(),
            );
            idx_pb.update(|x| x.set_pos(idx as u64));
        } else {
            let word_to_place =
                ss.words_per_column[vec_idx.as_slice()].as_ref().unwrap()[*iter_idx].clone();
            // println!("Placing word: {} at index {}", word_to_place.iter().collect::<String>(), idx);
            iter::zip(
                ss.puzzle
                    .slice_mut::<SliceInfo<_, IxDyn, IxDyn>>(column_slice_info_builder(
                        vec_idx,
                        length as isize,
                    ))
                    .iter_mut(),
                word_to_place.iter(),
            )
            .for_each(|(a, b)| *a = *b);

            ss.used_words.insert(word_to_place);

            *ss.iters[index_to_column(idx, length, dims).as_slice()]
                .as_mut()
                .unwrap() += 1;
            iter_pbs[idx].inc(1);

            idx += 1;
            idx_pb.update(|x| x.set_pos(idx as u64));
        }

        // If we have successfully placed every word, print and exit
        if idx == (length as usize).pow(dims as u32 - 1) {
            println!("Success");
            println!("{:?}", ss.puzzle);
            return;
        }
    }
}
