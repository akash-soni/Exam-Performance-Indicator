# End to End Machine Learning Project

# ML-project-template

ML project template

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```

```bash
conda activate ./env
```

OR

```bash
source activate ./env
```

### STEP 04- install the requirements

```bash
pip install -r requirements.txt
```

### STEP 05 - Create conda.yaml file -

```bash
conda env export > conda.yaml
```

### STEP 06 - Create Mkdocs pages

- In docs/ directory create "document.md"
- Once document is created add it to mkdocs.yml

```bash
nav:
  - Home: index.md
  - Page1: Page01.md
  - Page2: document.md
```

- Commit your code.
- Go to actions, check build and find the URL
- Copy the URL and go to About in Code section, Click on Settings icon and paste the URL under website tab.

### STEP 06- commit and push the changes to the remote repository
