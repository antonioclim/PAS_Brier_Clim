# `generate_stem_sim.py`: Synthetic Test Generation for Probabilistic Assessment

**Overview:** This Python script generates synthetic multiple-choice tests and associated probabilistic answer data for research in student assessment. It is designed to simulate **STEM** (science, technology, engineering, mathematics) quizzes where each question has several options and an objectively defined correct answer. The output includes a set of **probabilistic truth vectors** (one per question, encoding the correct answer as a probability distribution) and corresponding **belief distributions** for several hypothetical student profiles (archetypes). This allows researchers to explore how different types of test-takers (e.g. confident experts, uninformed guessers, partially knowledgeable students) would respond under a probabilistic scoring scheme. By controlling random seeds and parameters, the script ensures reproducible generation of test data, facilitating rigorous simulation studies and validation of scoring algorithms.

## Running the Script on Windows 11

You can execute `generate_stem_sim.py` on Windows 11 either using PowerShell or within Visual Studio Code’s integrated terminal. In all cases, ensure you have a suitable Python environment set up (Python 3.x installed and added to your PATH).

### Using PowerShell

1. **Open PowerShell:** Click the Start menu, search for “PowerShell”, and launch **Windows PowerShell** (or the newer **Windows Terminal**).  
2. **Navigate to the script directory:** Use the `cd` command to change to the folder containing `generate_stem_sim.py`. For example:  
   ```powershell
   cd C:\Users\YourName\Research\ProbabilisticAssessment\
   ```  
   (Replace the path with the actual directory where the script resides.)  
3. **Run the script with Python:** Invoke the script by typing the Python command followed by the script name and any desired options. For instance:  
   ```powershell
   python generate_stem_sim.py --num_tests 5 --num_questions 20 --seed 42 --output demo_output.csv
   ```  
   This command would generate 5 synthetic tests of 20 questions each, using a random seed of 42, and save the output to `demo_output.csv`. If you omit options, the script will use its built-in default parameters. You can also run `python generate_stem_sim.py -h` to display a help message with all available options and default values.  
4. **Observe output:** Once executed, the script will generate the specified output file in the current directory (or at the given `--output` path). For example, after running the above command, you should find `demo_output.csv` (or a zipped file, as explained below) in the directory. The script may also print a summary of what was generated (e.g., number of tests, questions, etc.) to the console for verification.  
5. **(Optional) Zip the results:** If you included the `--zip` flag in the command, the script will compress the output file. For example, using `--output demo_output.csv --zip` will produce a file `demo_output.zip` containing the CSV. This is useful when the output is large or if you intend to share the results as a single archive.

### Using Visual Studio Code

1. **Open the project in VS Code:** Launch **Visual Studio Code** and open the folder containing `generate_stem_sim.py` (via *File → Open Folder…*). Ensure the Python extension is installed and that VS Code is using the correct Python interpreter for your project.  
2. **Open a terminal in VS Code:** Go to *Terminal → New Terminal* to open an integrated PowerShell terminal (or Command Prompt) at the project directory. The terminal should already be at the correct path if you opened the folder in the previous step.  
3. **Run the script from the terminal:** In the VS Code terminal, execute the script with Python just as you would in a standalone PowerShell. For example:  
   ```powershell
   python generate_stem_sim.py --num_tests 1 --num_questions 10 --output test_simulation.json
   ```  
   This will run the simulation for 1 test of 10 questions and output the data to `test_simulation.json`. You will see any console output (status messages, etc.) directly in the VS Code terminal.  
4. **Alternative – Use the Run button:** As an alternative to using the terminal, you can run the script by opening `generate_stem_sim.py` in the editor and clicking the “Run Python File” button (▶️) provided by the Python extension. This will execute the script with default settings (equivalent to running without command-line arguments). The output file will still be produced as usual in the workspace folder.  
5. **Verify output in VS Code:** After execution, you can refresh the VS Code file explorer to see the generated output file (e.g., CSV or JSON). You may open this file in VS Code to inspect its contents (for instance, to verify the format of the truth vectors and belief distributions). If the `--zip` option was used, ensure you locate the ZIP archive; you can open it with VS Code or a zip utility to see the enclosed data file.

## Configuring Script Parameters

The `generate_stem_sim.py` script exposes several command-line parameters to customize the simulation. By adjusting these, you can control the size and randomness of the generated dataset to fit your experimental needs:

- **Number of tests (`--num_tests`):** Sets how many independent test instances (quiz forms or datasets) to generate. For example, `--num_tests 10` will create ten separate synthetic tests. Each test will have its own set of questions and corresponding data. *(Default: 1 test, unless otherwise specified in the script.)*

- **Number of questions (`--num_questions`):** Sets the number of multiple-choice questions per test. For example, `--num_questions 50` generates a 50-question assessment (each test will contain 50 items). All generated questions will follow the multiple-choice format with a fixed number of options (e.g., 4 options per question, as defined in the script’s internal settings). *(Default: typically 10 or another reasonable number if not provided.)*

- **Random seed (`--seed`):** An integer seed for the random number generator to ensure reproducibility. Providing a seed guarantees that the same “random” test data is generated every run, which is critical for reproducible research. For instance, using `--seed 12345` will always produce the identical set of tests and distributions on each run. If no seed is specified, the script may use a default seed or a system time-based seed (leading to different outputs each run). It is good practice to set this for experiments you plan to share or compare.

- **Output file name (`--output`):** The file path or name for the output data. The script will save the generated test dataset to this file. You can specify a relative or absolute path. For example, `--output sim_data.json` writes the results to a JSON file named *sim_data.json* in the current directory. The format of the output can be inferred from the file extension (e.g., use `.csv` for a comma-separated values file or `.json` for a JSON structured file) – consult the script documentation to see which formats are supported. *(Default: if not given, the script may use a default filename, such as `stem_sim_output.csv`, in the working directory.)*

- **ZIP compression (`--zip` flag):** Include this flag (with no additional value) if you wish to compress the output. When `--zip` is enabled, the script will create a ZIP archive containing the output file instead of leaving the raw output file on disk. For example, `--output sim_data.csv --zip` will result in `sim_data.zip` (containing the CSV file). This feature is useful for bundling results (especially if `--num_tests` or `--num_questions` are large, producing a large data file) and for preparing the simulation output for sharing or publication as supplementary material. *(Default: off; no compression unless explicitly requested.)*

**Example usage:** To illustrate, the following command generates 3 tests each with 30 questions, using a fixed seed of 2025, writing the output to `experiment_dataset.csv` and compressing it:

