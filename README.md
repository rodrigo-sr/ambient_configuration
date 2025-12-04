# Linux Data Science & Gaming Setup Script

This Python script automates the complete setup of a fresh Linux installation for **Data Science**, **Machine Learning**, and **Gaming** with NVIDIA GPU support.

## Features

### Core Development Environment
- **Essential tools**: Git, Vim, Neovim, tmux, zsh, fish, htop, ripgrep, fzf
- **Python**: Python 3, pip, venv, poetry, black, flake8, mypy, pytest, ruff
- **Node.js**: LTS version with npm, yarn, pnpm (for dashboards)
- **Docker**: Docker CE with Docker Compose
- **VS Code**: Visual Studio Code editor
- **Miniconda**: Environment management for data science projects

### Data Science Stack
- **Core Libraries**: NumPy, Pandas, Polars, SciPy, PyArrow
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost, StatsModels
- **Visualization**: Matplotlib, Seaborn, Plotly, Altair
- **Data Processing**: DuckDB, FastParquet, OpenPyXL

### Data Engineering
- **Big Data**: PySpark, Delta-Spark, Apache Kafka
- **Orchestration**: Apache Airflow, Prefect, Dagster
- **Cloud Storage**: boto3, google-cloud-storage, azure-storage-blob, MinIO
- **Task Queues**: Celery, Redis

### Databases
- **PostgreSQL**, **MySQL**, **MongoDB** (local installations)
- **DBeaver** - Universal database client
- Python connectors: SQLAlchemy, psycopg2, pymysql

### R Environment
- **R** with tidyverse, ggplot2, dplyr, caret
- **R kernel** for Jupyter notebooks
- Common data science packages pre-installed

### Gaming Setup
- **NVIDIA Drivers**: Latest proprietary drivers with CUDA support
- **Steam**: Full Steam client with Vulkan support
- **Proton-GE**: Custom Proton for better Windows game compatibility
- **GameMode**: Automatic performance optimization during gaming
- **MangoHud**: In-game performance overlay (FPS, temps, usage)
- **System Tweaks**: Optimized kernel parameters for gaming

## Supported Distributions

- Ubuntu / Debian / Pop!_OS / Linux Mint
- Fedora / RHEL / CentOS
- Arch Linux / Manjaro / EndeavourOS

## Requirements

- Fresh Linux installation
- Internet connection
- NVIDIA GPU (for gaming features)
- Root/sudo access

## Installation

### Quick Install

```bash
cd /home/rodrigo/Dev/linux-dev-setup
source venv/bin/activate
sudo python3 setup_dev_environment.py
```

### Step by Step

1. Clone or navigate to this directory:
   ```bash
   cd /home/rodrigo/Dev/linux-dev-setup
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Run the setup script with sudo:
   ```bash
   sudo python3 setup_dev_environment.py
   ```

4. **Reboot** your system after the script completes to load NVIDIA drivers.

## Post-Installation Steps

1. **Reboot** your system to load NVIDIA drivers properly

2. **Restart terminal** or run to activate Conda:
   ```bash
   source ~/.bashrc
   # or for fish
   source ~/.config/fish/config.fish
   ```

3. **Configure Git** with your personal information:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your@email.com"
   ```

4. **Start Jupyter Lab**:
   ```bash
   jupyter lab
   ```

5. **Create a Conda environment** for your project:
   ```bash
   conda create -n myproject python=3.11
   conda activate myproject
   pip install pandas numpy scikit-learn matplotlib
   ```

6. **Test PySpark**:
   ```bash
   python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.getOrCreate(); print('Spark version:', spark.version)"
   ```

7. **Steam Setup** (for gaming):
   - Open Steam and log in
   - Go to Settings > Compatibility
   - Enable "Enable Steam Play for all other titles"
   - Select Proton-GE as the default compatibility tool

## Configuration Files Created

| File | Purpose |
|------|---------|
| `~/miniconda3/` | Miniconda installation |
| `~/.config/MangoHud/MangoHud.conf` | MangoHud overlay settings |
| `~/.config/nvidia/gaming-config.sh` | NVIDIA gaming optimizations |
| `/etc/X11/xorg.conf.d/20-nvidia.conf` | NVIDIA Xorg configuration |
| `/etc/sysctl.d/99-gaming.conf` | Kernel parameters for gaming |

## Quick Commands Cheat Sheet

```bash
# Create new conda environment
conda create -n myenv python=3.11
conda activate myenv

# Install packages in environment
pip install pandas numpy scikit-learn

# Start PySpark shell
pyspark

# Start Airflow
airflow standalone

# Check GPU status
nvidia-smi

# Run Python with GPU monitoring
watch -n 1 nvidia-smi
```

## Customization

You can modify the script to customize:

- **Data Science packages**: Edit `install_data_science_stack()`
- **Data Engineering tools**: Edit `install_data_engineering_tools()`
- **R packages**: Edit `install_r_environment()`
- **Database tools**: Edit `install_database_tools()`
- **MangoHud**: Edit `~/.config/MangoHud/MangoHud.conf` after installation

## Troubleshooting

### NVIDIA/CUDA Issues
```bash
# Check if NVIDIA module is loaded
lsmod | grep nvidia

# Check NVIDIA driver status
nvidia-smi

# Check CUDA version
nvcc --version
```

### Conda Issues
```bash
# Initialize conda for your shell
conda init bash  # or fish, zsh

# Update conda
conda update conda

# Clean conda cache
conda clean --all
```

## License

This script is provided as-is for personal use. Feel free to modify and distribute.

## Contributing

Suggestions and improvements are welcome! The script is designed to be modular, making it easy to add support for additional distributions or tools.
