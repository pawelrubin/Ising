use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use console::Emoji;
use linya::{Bar, Progress};
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

const MIN_TEMP: f64 = 1.0;
const MAX_TEMP: f64 = 5.0;
const TEMP_STEP: f64 = 0.05;
const INITIAL_STEPS: u32 = 30_000;
const LATER_STEPS: u32 = 200_000;
const MAGN_CALC_STEP: u32 = 100;
const MAGN_STEPS: u32 = (LATER_STEPS / MAGN_CALC_STEP) as u32;
const LATTICE_SIZES: [usize; 4] = [6, 15, 40, 70];

static SPARKLE: Emoji<'_, '_> = Emoji("✨", ":)");
static ROCKET: Emoji<'_, '_> = Emoji("🚀", ":o");

fn generate_lattice(size: usize) -> Vec<i8> {
    (0..size * size)
        .map(|_| if rand::random() { 1 } else { -1 })
        .collect()
}

fn get_adjacent_indices(index: usize, size: usize) -> [usize; 4] {
    /*
    Returns adjacent indices on a torus,
    represented in a vector, in the following form:
    [left, top, right, bottom]
    */
    let bottom_left = size * size - size;
    [
        if index % size == 0 {
            // left side
            index + size - 1
        } else {
            index - 1
        },
        if index < size {
            // top side
            index + bottom_left
        } else {
            index - size
        },
        if index % size == size - 1 {
            // right side
            index + 1 - size
        } else {
            index + 1
        },
        if index >= bottom_left {
            // bottom side
            index - bottom_left
        } else {
            index + size
        },
    ]
}

fn get_trans_map(temp: f64) -> HashMap<i8, f64> {
    [
        (-8, (8.0 / temp).exp().min(1.0)),
        (-4, (4.0 / temp).exp().min(1.0)),
        (0, 1.0),
        (4, (-4.0 / temp).exp().min(1.0)),
        (8, (-8.0 / temp).exp().min(1.0)),
    ]
    .iter()
    .cloned()
    .collect()
}

fn recalc_lattice(lattice: &mut Vec<i8>, size: usize, trans_map: &HashMap<i8, f64>) {
    for index in 0..size * size {
        let spin = lattice[index];
        let energy_change = 2
            * spin
            * get_adjacent_indices(index, size)
                .iter()
                .map(|i| lattice[*i])
                .sum::<i8>();
        let mut rng = rand::thread_rng();
        if rng.gen_bool(trans_map[&energy_change]) {
            lattice[index] = -spin;
        }
    }
}

fn get_magnetization(lattice: &Vec<i8>) -> f64 {
    (lattice.par_iter().map(|e| *e as i32).sum::<i32>() as f64 / lattice.len() as f64).abs()
}

fn get_float_range(start: f64, end: f64, step: f64) -> Vec<f64> {
    ((start * 100.0) as i32..(end * 100.0) as i32)
        .step_by((step * 100.0) as usize)
        .map(|x| x as f64 * 0.01)
        .collect()
}

fn get_params() -> Vec<(usize, f64, HashMap<i8, f64>)> {
    let temperatures = get_float_range(MIN_TEMP, MAX_TEMP, TEMP_STEP);
    LATTICE_SIZES
        .iter()
        .flat_map(|lattice_size| {
            temperatures.iter().map(move |temp| {
                let trans_map = get_trans_map(*temp);
                (*lattice_size, *temp, trans_map)
            })
        })
        .collect()
}

fn iteration(lattice_size: usize, temperature: f64, trans_map: &HashMap<i8, f64>) -> (f64, f64) {
    let mut lattice = generate_lattice(lattice_size);

    (0..INITIAL_STEPS).for_each(|_| {
        recalc_lattice(&mut lattice, lattice_size, &trans_map);
    });

    let (magn_sum, magn_sqrt_sum) = (0..LATER_STEPS).into_iter().fold((0.0, 0.0), |acc, i| {
        recalc_lattice(&mut lattice, lattice_size, &trans_map);
        if i % MAGN_CALC_STEP == 0 {
            let current_magnetization = get_magnetization(&lattice);

            return (
                acc.0 + current_magnetization,
                acc.1 + current_magnetization * current_magnetization,
            );
        }
        return acc;
    });

    let magnetization = magn_sum / MAGN_STEPS as f64;
    let susceptibility = ((lattice_size * lattice_size) as f64 / temperature)
        * (magn_sqrt_sum / MAGN_STEPS as f64 - magnetization * magnetization);
    (magnetization, susceptibility)
}

fn main() {
    let started = Instant::now();

    // get parameters for the simulations
    let params: Vec<(usize, f64, HashMap<i8, f64>)> = get_params();
    let progress = Arc::new(Mutex::new(Progress::new()));
    let bar: Bar = progress
        .lock()
        .unwrap()
        .bar(params.len(), "Running simulations");

    // run simulations in parallel
    let results =
        params
            .par_iter()
            .map_with(progress, |p, (lattice_size, temperature, trans_map)| {
                let (magnetization, susceptibility) =
                    iteration(*lattice_size, *temperature, trans_map);

                p.lock().unwrap().inc_and_draw(&bar, 1);
                (*lattice_size, *temperature, magnetization, susceptibility)
            });

    // write the results
    let output_file = Arc::new(Mutex::new(File::create("ising.txt").unwrap()));
    writeln!(output_file.lock().unwrap(), "l t m s").unwrap();
    results.for_each(|result| {
        let (l, t, m, s) = result;
        writeln!(
            output_file.lock().unwrap(),
            "{}",
            format!("{} {:.2} {:.5} {:.5}", l, t, m, s)
        )
        .unwrap();
    });

    println!("{} Done in {:?} {}", SPARKLE, started.elapsed(), ROCKET);
}
