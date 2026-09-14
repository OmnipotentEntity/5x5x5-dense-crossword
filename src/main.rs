// use std::cmp::Reverse;
use std::collections::hash_set::Iter;
use std::collections::{BinaryHeap, HashSet};
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::iter;
use std::path::Path;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::{s, Array, Array2, Axis, IxDyn, SliceInfo, SliceInfoElem};
use ouroboros::self_referencing;

const EMPTY_CELL: char = ' ';
const DEFAULT_DIMS: usize = 3;
const DEFAULT_LENGTH: usize = 5;

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

fn build_prefix_set(words: Array2<char>, length: usize) -> HashSet<Vec<char>> {
    let mut result = HashSet::<Vec<char>>::new();
    for i in 1..=length {
        for row in words.rows().into_iter() {
            let key_vec = row
                .slice(s![0..i])
                .to_owned()
                .into_iter()
                .collect::<Vec<_>>();
            result.insert(key_vec);
        }
    }

    result
}

#[self_referencing]
struct SolverState {
    // These are all of the valid words
    words: HashSet<Vec<char>>,

    // These are the pared down sets of words for each column
    words_per_column : Array<Option<HashSet<Vec<char>>>, IxDyn>,

    // These are all of the words used so far
    used_words: HashSet<Vec<char>>,

    // This field is for backtracking, for each entry in the working dimension, we try to place a
    // word, and if we fail then we jump back to a previous word.  These iters are how much we've
    // tried so far.
    #[borrows(mut words_per_column)]
    #[covariant]
    iters: Array<Option<Iter<'this, Vec<char>>>, IxDyn>,

    // These are words that we've previously calculated to conflict along a given axis, and this
    // axis hasn't changed since the last time we visited this column
    conflicts: Array<HashSet<Vec<char>>, IxDyn>,

    // The working output of the solver.  A value of EMPTY_CELL represents unset.
    puzzle: Array<char, IxDyn>,
}

impl SolverState {
    fn init(words: Array2<char>, length: usize, dims: usize) -> SolverState {
        SolverStateBuilder {
            words: HashSet::from_iter(words.axis_iter(Axis(0)).map(|x| x.to_vec())),
            words_per_column: 
                Array::from_iter(iter::repeat_n(None, length.pow(dims as u32 - 1)))
                    .into_shape_with_order(IxDyn(
                        Vec::from_iter(iter::repeat_n(length, dims - 1)).as_slice(),
                    ))
                    .unwrap(),
            used_words: HashSet::new(),
            iters_builder: |_| {
                Array::from_iter(iter::repeat_n(None, length.pow(dims as u32 - 1)))
                    .into_shape_with_order(IxDyn(
                        Vec::from_iter(iter::repeat_n(length, dims - 1)).as_slice(),
                    ))
                    .unwrap()
            },
            conflicts: Array::from_iter(iter::repeat_n(
                HashSet::new(),
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
        .build()
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

    let mut solver_state = SolverState::init(valid_words, length.into(), dims.into());

    solver_state.with_mut(|ss| {
        let multibar = MultiProgress::new();
        let mut pbs = Vec::from_iter(iter::repeat_n(
            Option::<ProgressBar>::None,
            length.pow(dims as u32 - 1),
        ));
        let mut idx = 0;
        loop {
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

            ss.words_per_column[vec_idx.as_slice()] = Some(&*ss.words - &*ss.used_words);
            for axis in 0..dims - 1 {
            }

            if pbs[idx].is_none() {
                let pb = multibar
                    .add(ProgressBar::new(ss.words.len() as u64).with_prefix(format!("{:?}", idx)));
                pb.set_style(ProgressStyle::with_template("{prefix} {bar} {pos}/{len}").unwrap());
                pbs[idx] = Some(pb);
            }

            // let iter = &mut ss.iters[vec_idx.as_slice()];
            // let mut placed_word = false;
            // let mut conflicting_axes: HashSet<usize> = HashSet::from_iter(0..dims as usize);

            // while let &Some(word) = &iter.next() {
            //     // Update progress bar
            //     pbs[idx].as_ref().unwrap().inc(1);

            //     // check prefixes
            //     let mut all_prefixes_valid = true;
            //     let mut this_conflicting_axes: HashSet<usize> = HashSet::new();
            //     for i in 0..dims as usize - 1 {
            //         for j in 0..length as usize {
            //             let mut puzzle_prefix_for_dim_i = Vec::from_iter(
            //                 ss.puzzle
            //                     .slice(slice_info_builder(i, j, vec_idx.clone()))
            //                     .to_owned()
            //                     .into_iter(),
            //             );
            //             puzzle_prefix_for_dim_i.push(word[j]);

            //             if prefixes.get(&puzzle_prefix_for_dim_i).is_none() {
            //                 all_prefixes_valid = false;
            //                 this_conflicting_axes.insert(i);
            //                 break;
            //             }
            //         }
            //     }

            //     // if fine place word
            //     if all_prefixes_valid {
            //         let mut slice_info = vec_idx
            //             .iter()
            //             .map(|x| SliceInfoElem::Index(*x as isize))
            //             .collect::<Vec<SliceInfoElem>>();
            //         slice_info.push(SliceInfoElem::Slice {
            //             start: 0,
            //             end: Some(length as isize),
            //             step: 1,
            //         });
            //         iter::zip(
            //             ss.puzzle
            //                 .slice_mut::<SliceInfo<_, IxDyn, IxDyn>>(
            //                     SliceInfo::try_from(slice_info).unwrap(),
            //                 )
            //                 .iter_mut(),
            //             word.iter(),
            //         )
            //         .for_each(|(a, b)| *a = *b);
            //         placed_word = true;
            //         break;
            //     } else {
            //         // otherwise, update the current list of conflicted axes
            //         conflicting_axes = &conflicting_axes & &this_conflicting_axes;
            //     }
            // }

            // if placed_word {
            //     idx += 1;

            //     // If we have successfully placed every word, print and exit
            //     if idx == (length as usize).pow(dims as u32 - 1) {
            //         pbs.iter_mut().for_each(|x| {
            //             x.as_ref().map(|x| x.finish());
            //         });
            //         println!("Success");
            //         println!("{:?}", ss.puzzle);
            //         return;
            //     }

            //     // reset iterators when moving forwards
            //     // (Currently broken due to an aborted rewrite of this function)
            //     // ss.words_per_column[index_to_column(idx, length.into(), dims.into()).as_slice()] = Some(&ss.words - &ss.words_used - 
            //     // ss.iters[index_to_column(idx, length.into(), dims.into()).as_slice()] =
            //     //     ss.words.iter();
            // } else {
            //     // If we have exhausted all possible words, then we error and exit
            //     if idx == 0 {
            //         println!("Failure");
            //         return;
            //     }

            //     // Otherwise roll back along the least significant conflicting axis
            //     let rollback_dim = if conflicting_axes.is_empty() {
            //         0
            //     } else {
            //         let mut dim_heap =
            //             BinaryHeap::from_iter(conflicting_axes.into_iter().map(|x| Reverse(x)));
            //         dim_heap.pop().unwrap().0
            //     };
            //     // println!("rollback dim: {:?}", rollback_dim);
            //     let mut vec_idx = index_to_column(idx, length.into(), dims.into());
            //     vec_idx[rollback_dim] -= 1;

            //     let prev_idx = idx;
            //     idx = column_to_index(vec_idx, length.into());
            //     // println!("rollback to idx: {:?}", idx);

            //     // clear the progress bars
            //     pbs[idx + 1..=prev_idx].iter_mut().for_each(|x| *x = None);
            // }
        }
    });
}
