# Ascii-Based-Filter

This program creates ASCII art from images by converting luminance to characters and edges to directional symbols. The current implementation replaces luminance values with ASCII characters (.,c,p,O), with edge detection and directional character replacement as a work-in-progress feature.

## Features
- Luminance to ASCII conversion: Replaces pixel brightness with ASCII characters
- Edge detection: Identifies edges using Sobel filter

## Installation
1. Clone the repository:
```bash
git clone https://github.com/vader-pepe/ascii-based-filter.git
cd ascii-based-filter
```
2. Build the project:
```bash
cargo build --release
```

## Usage
```bash
./target/release/asbf input.jpg
```

## Example Output
![Example1](https://github.com/vader-pepe/ascii-based-filter/blob/main/results/c1.png)

## Contributing
Contributions are welcome! Please focus on:

1. Completing the edge-to-character replacement
2. Improving edge detection accuracy
3. Optimizing performance
4. Use GPU
5. Adding test cases


