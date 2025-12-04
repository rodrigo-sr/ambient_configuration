#!/usr/bin/env python3
"""
Linux Data Science & Development Environment Setup Script

This script automates the configuration of a fresh Linux installation for:
- Data Science and Data Analytics tools
- Python data science stack (Pandas, NumPy, Scikit-learn, etc.)
- Machine Learning frameworks (TensorFlow, PyTorch)
- Jupyter ecosystem and notebooks
- Database clients and tools
- NVIDIA GPU drivers for ML/AI workloads
- Steam gaming with NVIDIA GPU support

Supports: Ubuntu/Debian, Fedora, Arch Linux

Run with: sudo python3 setup_dev_environment.py
"""

import subprocess
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Callable
from enum import Enum, auto


class Distro(Enum):
    UBUNTU_DEBIAN = auto()
    FEDORA = auto()
    ARCH = auto()
    UNKNOWN = auto()


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(message: str) -> None:
    """Print a formatted header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_step(message: str) -> None:
    """Print a step message"""
    print(f"{Colors.CYAN}>>> {message}{Colors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def run_command(command: str, shell: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Execute a shell command and return the result"""
    print_step(f"Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {e}")
        if e.stderr:
            print(e.stderr)
        raise


def detect_distro() -> Distro:
    """Detect the Linux distribution"""
    print_step("Detecting Linux distribution...")
    
    if Path("/etc/os-release").exists():
        with open("/etc/os-release") as f:
            content = f.read().lower()
            
        if "ubuntu" in content or "debian" in content or "pop" in content or "mint" in content:
            print_success("Detected: Ubuntu/Debian-based distribution")
            return Distro.UBUNTU_DEBIAN
        elif "fedora" in content or "rhel" in content or "centos" in content:
            print_success("Detected: Fedora/RHEL-based distribution")
            return Distro.FEDORA
        elif "arch" in content or "manjaro" in content or "endeavour" in content:
            print_success("Detected: Arch-based distribution")
            return Distro.ARCH
    
    print_warning("Could not detect distribution")
    return Distro.UNKNOWN


def check_root() -> bool:
    """Check if script is running as root"""
    return os.geteuid() == 0


def update_system(distro: Distro) -> None:
    """Update system packages"""
    print_header("Updating System Packages")
    
    commands = {
        Distro.UBUNTU_DEBIAN: "apt update && apt upgrade -y",
        Distro.FEDORA: "dnf update -y",
        Distro.ARCH: "pacman -Syu --noconfirm",
    }
    
    if distro in commands:
        run_command(commands[distro])
        print_success("System updated successfully")
    else:
        print_warning("Skipping system update - unknown distribution")


def install_essential_packages(distro: Distro) -> None:
    """Install essential development packages"""
    print_header("Installing Essential Packages")
    
    # Common packages across distributions
    common_packages = {
        Distro.UBUNTU_DEBIAN: [
            "build-essential", "git", "curl", "wget", "vim", "neovim",
            "htop", "tree", "unzip", "zip", "software-properties-common",
            "apt-transport-https", "ca-certificates", "gnupg", "lsb-release",
            "net-tools", "openssh-server", "tmux", "zsh", "fish",
            "jq", "ripgrep", "fd-find", "bat", "exa", "fzf"
        ],
        Distro.FEDORA: [
            "gcc", "gcc-c++", "make", "git", "curl", "wget", "vim", "neovim",
            "htop", "tree", "unzip", "zip", "dnf-plugins-core",
            "net-tools", "openssh-server", "tmux", "zsh", "fish",
            "jq", "ripgrep", "fd-find", "bat", "exa", "fzf"
        ],
        Distro.ARCH: [
            "base-devel", "git", "curl", "wget", "vim", "neovim",
            "htop", "tree", "unzip", "zip", "net-tools", "openssh",
            "tmux", "zsh", "fish", "jq", "ripgrep", "fd", "bat", "exa", "fzf"
        ],
    }
    
    install_commands = {
        Distro.UBUNTU_DEBIAN: "apt install -y",
        Distro.FEDORA: "dnf install -y",
        Distro.ARCH: "pacman -S --noconfirm",
    }
    
    if distro in common_packages:
        packages = " ".join(common_packages[distro])
        run_command(f"{install_commands[distro]} {packages}", check=False)
        print_success("Essential packages installed")


def install_python_dev(distro: Distro) -> None:
    """Install Python development tools"""
    print_header("Installing Python Development Environment")
    
    packages = {
        Distro.UBUNTU_DEBIAN: [
            "python3", "python3-pip", "python3-venv", "python3-dev",
            "python3-setuptools", "python3-wheel", "pipx"
        ],
        Distro.FEDORA: [
            "python3", "python3-pip", "python3-devel",
            "python3-setuptools", "python3-wheel", "pipx"
        ],
        Distro.ARCH: [
            "python", "python-pip", "python-setuptools", "python-wheel", "python-pipx"
        ],
    }
    
    install_commands = {
        Distro.UBUNTU_DEBIAN: "apt install -y",
        Distro.FEDORA: "dnf install -y",
        Distro.ARCH: "pacman -S --noconfirm",
    }
    
    if distro in packages:
        pkg_list = " ".join(packages[distro])
        run_command(f"{install_commands[distro]} {pkg_list}", check=False)
    
    # Install common Python tools via pip
    python_tools = ["black", "flake8", "mypy", "pytest", "poetry", "ipython", "ruff"]
    for tool in python_tools:
        run_command(f"pip3 install --user {tool}", check=False)
    
    print_success("Python development environment installed")


def install_nodejs(distro: Distro) -> None:
    """Install Node.js and npm"""
    print_header("Installing Node.js Development Environment")
    
    # Install using NodeSource for latest LTS version
    if distro == Distro.UBUNTU_DEBIAN:
        run_command("curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -", check=False)
        run_command("apt install -y nodejs", check=False)
    elif distro == Distro.FEDORA:
        run_command("dnf install -y nodejs npm", check=False)
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm nodejs npm", check=False)
    
    # Install common global npm packages
    npm_packages = ["typescript", "ts-node", "eslint", "prettier", "yarn", "pnpm"]
    for pkg in npm_packages:
        run_command(f"npm install -g {pkg}", check=False)
    
    print_success("Node.js development environment installed")


def install_rust(distro: Distro) -> None:
    """Install Rust programming language"""
    print_header("Installing Rust Development Environment")
    
    # Install rustup
    run_command("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", check=False)
    
    # Source cargo environment
    cargo_env = Path.home() / ".cargo" / "env"
    if cargo_env.exists():
        run_command(f"source {cargo_env} && rustup component add clippy rustfmt", check=False)
    
    print_success("Rust development environment installed")


def install_go(distro: Distro) -> None:
    """Install Go programming language"""
    print_header("Installing Go Development Environment")
    
    if distro == Distro.UBUNTU_DEBIAN:
        run_command("apt install -y golang-go", check=False)
    elif distro == Distro.FEDORA:
        run_command("dnf install -y golang", check=False)
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm go", check=False)
    
    print_success("Go development environment installed")


def install_docker(distro: Distro) -> None:
    """Install Docker and Docker Compose"""
    print_header("Installing Docker")
    
    if distro == Distro.UBUNTU_DEBIAN:
        # Add Docker's official GPG key
        run_command("install -m 0755 -d /etc/apt/keyrings", check=False)
        run_command("curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg", check=False)
        run_command("chmod a+r /etc/apt/keyrings/docker.gpg", check=False)
        
        # Add the repository
        run_command('''echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null''', check=False)
        
        run_command("apt update", check=False)
        run_command("apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin", check=False)
        
    elif distro == Distro.FEDORA:
        run_command("dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo", check=False)
        run_command("dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin", check=False)
        
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm docker docker-compose", check=False)
    
    # Start and enable Docker
    run_command("systemctl start docker", check=False)
    run_command("systemctl enable docker", check=False)
    
    # Add current user to docker group
    user = os.environ.get("SUDO_USER", os.environ.get("USER"))
    if user:
        run_command(f"usermod -aG docker {user}", check=False)
    
    print_success("Docker installed and configured")


def install_vscode(distro: Distro) -> None:
    """Install Visual Studio Code"""
    print_header("Installing Visual Studio Code")
    
    if distro == Distro.UBUNTU_DEBIAN:
        run_command("wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg", check=False)
        run_command("install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg", check=False)
        run_command('''echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | tee /etc/apt/sources.list.d/vscode.list > /dev/null''', check=False)
        run_command("rm -f packages.microsoft.gpg", check=False)
        run_command("apt update", check=False)
        run_command("apt install -y code", check=False)
        
    elif distro == Distro.FEDORA:
        run_command("rpm --import https://packages.microsoft.com/keys/microsoft.asc", check=False)
        run_command('''echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" | tee /etc/yum.repos.d/vscode.repo > /dev/null''', check=False)
        run_command("dnf check-update", check=False)
        run_command("dnf install -y code", check=False)
        
    elif distro == Distro.ARCH:
        # Install from official repos (code is available as 'code' package)
        run_command("pacman -S --noconfirm code", check=False)
    
    print_success("Visual Studio Code installed")


def install_discord(distro: Distro) -> None:
    """Install Discord chat application"""
    print_header("Installing Discord")
    
    if distro == Distro.UBUNTU_DEBIAN:
        # Download and install Discord .deb package
        run_command("wget -O /tmp/discord.deb 'https://discord.com/api/download?platform=linux&format=deb'", check=False)
        run_command("dpkg -i /tmp/discord.deb || apt-get -f install -y", check=False)
        run_command("rm -f /tmp/discord.deb", check=False)
        
    elif distro == Distro.FEDORA:
        # Install via Flatpak (most reliable for Fedora)
        run_command("dnf install -y flatpak", check=False)
        run_command("flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo", check=False)
        run_command("flatpak install -y flathub com.discordapp.Discord", check=False)
        
    elif distro == Distro.ARCH:
        # Discord is available in official repos
        run_command("pacman -S --noconfirm discord", check=False)
    
    print_success("Discord installed")


def install_nvidia_drivers(distro: Distro) -> None:
    """Install NVIDIA GPU drivers"""
    print_header("Installing NVIDIA GPU Drivers")
    
    # Check if NVIDIA GPU is present
    result = run_command("lspci | grep -i nvidia", check=False)
    if result.returncode != 0:
        print_warning("No NVIDIA GPU detected, skipping driver installation")
        return
    
    print_success("NVIDIA GPU detected")
    
    if distro == Distro.UBUNTU_DEBIAN:
        # Add graphics drivers PPA
        run_command("add-apt-repository -y ppa:graphics-drivers/ppa", check=False)
        run_command("apt update", check=False)
        
        # Install recommended driver
        run_command("ubuntu-drivers autoinstall", check=False)
        
        # Install additional NVIDIA packages
        run_command("apt install -y nvidia-driver-545 nvidia-settings nvidia-prime", check=False)
        
    elif distro == Distro.FEDORA:
        # Enable RPM Fusion repositories
        run_command("dnf install -y https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm", check=False)
        run_command("dnf install -y https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm", check=False)
        
        # Install NVIDIA drivers
        run_command("dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda", check=False)
        
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm nvidia nvidia-utils nvidia-settings lib32-nvidia-utils", check=False)
    
    print_success("NVIDIA drivers installed")


def configure_nvidia_for_gaming() -> None:
    """Configure NVIDIA settings optimized for gaming"""
    print_header("Configuring NVIDIA for Gaming")
    
    # Create NVIDIA settings configuration
    nvidia_config = """
# NVIDIA Gaming Configuration
# Enable SLI (if applicable)
# nvidia-settings -a "SLIMode=1"

# Force full composition pipeline for smoother gaming
nvidia-settings --assign CurrentMetaMode="nvidia-auto-select +0+0 { ForceFullCompositionPipeline = On }"

# Enable Coolbits for overclocking (optional)
# nvidia-xconfig --cool-bits=28
"""
    
    config_dir = Path.home() / ".config" / "nvidia"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "gaming-config.sh"
    with open(config_file, "w") as f:
        f.write(nvidia_config)
    
    config_file.chmod(0o755)
    
    # Create Xorg configuration for NVIDIA
    xorg_nvidia_config = '''Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    Option         "Coolbits" "28"
    Option         "TripleBuffer" "True"
    Option         "NoLogo" "True"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    Option         "metamodes" "nvidia-auto-select +0+0 { ForceFullCompositionPipeline = On }"
EndSection
'''
    
    xorg_dir = Path("/etc/X11/xorg.conf.d")
    if xorg_dir.exists():
        xorg_file = xorg_dir / "20-nvidia.conf"
        with open(xorg_file, "w") as f:
            f.write(xorg_nvidia_config)
        print_success("NVIDIA Xorg configuration created")
    
    print_success("NVIDIA gaming configuration completed")


def install_steam(distro: Distro) -> None:
    """Install Steam with all dependencies"""
    print_header("Installing Steam")
    
    if distro == Distro.UBUNTU_DEBIAN:
        # Enable 32-bit architecture
        run_command("dpkg --add-architecture i386", check=False)
        run_command("apt update", check=False)
        
        # Install Steam and dependencies
        run_command("apt install -y steam steam-devices libgl1-mesa-dri:i386", check=False)
        
        # Install Vulkan support
        run_command("apt install -y libvulkan1 libvulkan1:i386 vulkan-tools mesa-vulkan-drivers mesa-vulkan-drivers:i386", check=False)
        
    elif distro == Distro.FEDORA:
        # Enable RPM Fusion if not already enabled
        run_command("dnf install -y https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm", check=False)
        
        # Install Steam
        run_command("dnf install -y steam", check=False)
        
        # Install Vulkan support
        run_command("dnf install -y vulkan-loader vulkan-tools mesa-vulkan-drivers", check=False)
        
    elif distro == Distro.ARCH:
        # Enable multilib repository
        run_command("sed -i '/\\[multilib\\]/,/Include/s/^#//' /etc/pacman.conf", check=False)
        run_command("pacman -Syu --noconfirm", check=False)
        
        # Install Steam and dependencies
        run_command("pacman -S --noconfirm steam lib32-nvidia-utils lib32-mesa", check=False)
        
        # Install Vulkan support
        run_command("pacman -S --noconfirm vulkan-icd-loader lib32-vulkan-icd-loader vulkan-tools", check=False)
    
    print_success("Steam installed with gaming dependencies")


def install_proton_ge() -> None:
    """Install Proton-GE for better game compatibility"""
    print_header("Installing Proton-GE")
    
    # Create Steam compatibility tools directory
    compat_dir = Path.home() / ".steam" / "root" / "compatibilitytools.d"
    compat_dir.mkdir(parents=True, exist_ok=True)
    
    # Download latest Proton-GE
    print_step("Downloading latest Proton-GE...")
    run_command("curl -s https://api.github.com/repos/GloriousEggroll/proton-ge-custom/releases/latest | grep browser_download_url | cut -d '\"' -f 4 | grep '.tar.gz$' | head -1 | xargs wget -O /tmp/proton-ge.tar.gz", check=False)
    
    # Extract to compatibility tools directory
    run_command(f"tar -xzf /tmp/proton-ge.tar.gz -C {compat_dir}", check=False)
    run_command("rm /tmp/proton-ge.tar.gz", check=False)
    
    print_success("Proton-GE installed - Enable it in Steam > Settings > Compatibility")


def install_gamemode(distro: Distro) -> None:
    """Install GameMode for performance optimization"""
    print_header("Installing GameMode")
    
    if distro == Distro.UBUNTU_DEBIAN:
        run_command("apt install -y gamemode", check=False)
    elif distro == Distro.FEDORA:
        run_command("dnf install -y gamemode", check=False)
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm gamemode lib32-gamemode", check=False)
    
    print_success("GameMode installed - Games will automatically use it when supported")


def install_mangohud(distro: Distro) -> None:
    """Install MangoHud for in-game performance overlay"""
    print_header("Installing MangoHud")
    
    if distro == Distro.UBUNTU_DEBIAN:
        run_command("apt install -y mangohud", check=False)
    elif distro == Distro.FEDORA:
        run_command("dnf install -y mangohud", check=False)
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm mangohud lib32-mangohud", check=False)
    
    # Create MangoHud configuration
    mangohud_config = """# MangoHud Configuration
legacy_layout=false
gpu_stats
gpu_temp
gpu_core_clock
gpu_mem_clock
gpu_power
gpu_load_change
gpu_load_value=50,90
gpu_load_color=FFFFFF,FFAA7F,CC0000
cpu_stats
cpu_temp
cpu_power
cpu_mhz
cpu_load_change
core_load_change
cpu_load_value=50,90
cpu_load_color=FFFFFF,FFAA7F,CC0000
io_read
io_write
vram
ram
fps
engine_version
vulkan_driver
wine
frame_timing=1
frametime_color=00FF00,FFFF00,FF0000
position=top-left
background_alpha=0.4
font_size=24
toggle_hud=Shift_R+F12
"""
    
    mangohud_dir = Path.home() / ".config" / "MangoHud"
    mangohud_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = mangohud_dir / "MangoHud.conf"
    with open(config_file, "w") as f:
        f.write(mangohud_config)
    
    print_success("MangoHud installed - Use 'mangohud %command%' in Steam launch options")


def configure_gaming_tweaks() -> None:
    """Apply system tweaks for better gaming performance"""
    print_header("Applying Gaming Performance Tweaks")
    
    # Increase file descriptor limits
    limits_config = """# Gaming performance limits
* soft nofile 1048576
* hard nofile 1048576
"""
    
    limits_file = Path("/etc/security/limits.d/99-gaming.conf")
    with open(limits_file, "w") as f:
        f.write(limits_config)
    
    # Disable CPU mitigations for better performance (optional, security trade-off)
    # This is commented out by default as it's a security consideration
    # run_command("sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"mitigations=off /' /etc/default/grub", check=False)
    
    # Enable vm.max_map_count for games that need it
    sysctl_config = """# Gaming sysctl settings
vm.max_map_count = 2147483642
vm.swappiness = 10
"""
    
    sysctl_file = Path("/etc/sysctl.d/99-gaming.conf")
    with open(sysctl_file, "w") as f:
        f.write(sysctl_config)
    
    # Apply sysctl changes
    run_command("sysctl --system", check=False)
    
    print_success("Gaming performance tweaks applied")


def install_extra_dev_tools(distro: Distro) -> None:
    """Install additional development tools"""
    print_header("Installing Extra Development Tools")
    
    # Install database clients
    if distro == Distro.UBUNTU_DEBIAN:
        run_command("apt install -y postgresql-client mysql-client redis-tools sqlite3", check=False)
    elif distro == Distro.FEDORA:
        run_command("dnf install -y postgresql redis sqlite", check=False)
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm postgresql redis sqlite", check=False)
    
    print_success("Extra development tools installed")


def install_data_science_stack() -> None:
    """Install core data science Python packages"""
    print_header("Installing Data Science Stack")
    
    # Core data science packages
    core_packages = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "altair",
    ]
    
    # Statistical and ML packages
    ml_packages = [
        "scikit-learn",
        "statsmodels",
        "xgboost",
        "lightgbm",
        "catboost",
    ]
    
    # Data manipulation and processing
    data_packages = [
        "polars",
        "pyarrow",
        "fastparquet",
        "openpyxl",
        "xlrd",
        "sqlalchemy",
        "psycopg2-binary",
        "pymysql",
        "duckdb",
    ]
    
    # Install all packages
    all_packages = core_packages + ml_packages + data_packages
    packages_str = " ".join(all_packages)
    run_command(f"pip3 install --user {packages_str}", check=False)
    
    print_success("Data science stack installed")


def install_jupyter_ecosystem() -> None:
    """Install Jupyter notebooks and related tools"""
    print_header("Installing Jupyter Ecosystem")
    
    jupyter_packages = [
        "jupyter",
        "jupyterlab",
        "notebook",
        "ipywidgets",
        "jupyterlab-git",
        "jupyterlab-lsp",
        "python-lsp-server[all]",
        "nbconvert",
        "nbformat",
        "voila",
        "jupyter-dash",
    ]
    
    packages_str = " ".join(jupyter_packages)
    run_command(f"pip3 install --user {packages_str}", check=False)
    
    # Enable JupyterLab extensions
    run_command("jupyter labextension install @jupyter-widgets/jupyterlab-manager", check=False)
    
    print_success("Jupyter ecosystem installed")
    print_step("To start JupyterLab, run: jupyter lab")


def install_deep_learning_frameworks() -> None:
    """Install deep learning frameworks (TensorFlow, PyTorch)"""
    print_header("Installing Deep Learning Frameworks")
    
    # Check if NVIDIA GPU is available for CUDA support
    result = run_command("lspci | grep -i nvidia", check=False)
    has_nvidia = result.returncode == 0
    
    if has_nvidia:
        print_step("NVIDIA GPU detected - installing CUDA-enabled versions")
        
        # PyTorch with CUDA
        run_command("pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", check=False)
        
        # TensorFlow (auto-detects GPU)
        run_command("pip3 install --user tensorflow[and-cuda]", check=False)
    else:
        print_step("No NVIDIA GPU detected - installing CPU versions")
        
        # PyTorch CPU
        run_command("pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", check=False)
        
        # TensorFlow CPU
        run_command("pip3 install --user tensorflow", check=False)
    
    # Additional DL tools
    dl_tools = [
        "transformers",
        "datasets",
        "accelerate",
        "keras",
        "onnx",
        "onnxruntime",
    ]
    
    packages_str = " ".join(dl_tools)
    run_command(f"pip3 install --user {packages_str}", check=False)
    
    print_success("Deep learning frameworks installed")


def install_data_visualization_tools(distro: Distro) -> None:
    """Install data visualization and BI tools"""
    print_header("Installing Data Visualization Tools")
    
    # Python visualization packages
    viz_packages = [
        "bokeh",
        "holoviews",
        "hvplot",
        "panel",
        "streamlit",
        "dash",
        "gradio",
        "pygwalker",
    ]
    
    packages_str = " ".join(viz_packages)
    run_command(f"pip3 install --user {packages_str}", check=False)
    
    print_success("Data visualization tools installed")


def install_mlops_tools() -> None:
    """Install MLOps and experiment tracking tools"""
    print_header("Installing MLOps Tools")
    
    mlops_packages = [
        "mlflow",
        "wandb",
        "dvc",
        "great-expectations",
        "evidently",
        "prefect",
        "dagster",
    ]
    
    packages_str = " ".join(mlops_packages)
    run_command(f"pip3 install --user {packages_str}", check=False)
    
    print_success("MLOps tools installed")


def install_r_environment(distro: Distro) -> None:
    """Install R programming language and common packages"""
    print_header("Installing R Environment")
    
    if distro == Distro.UBUNTU_DEBIAN:
        # Add CRAN repository for latest R
        run_command("apt install -y --no-install-recommends software-properties-common dirmngr", check=False)
        run_command("wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc", check=False)
        run_command("add-apt-repository -y 'deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/'", check=False)
        run_command("apt update", check=False)
        run_command("apt install -y r-base r-base-dev", check=False)
        
    elif distro == Distro.FEDORA:
        run_command("dnf install -y R R-devel", check=False)
        
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm r", check=False)
    
    # Install common R packages
    r_packages_cmd = '''R -e "install.packages(c('tidyverse', 'ggplot2', 'dplyr', 'tidyr', 'readr', 'lubridate', 'stringr', 'devtools', 'shiny', 'rmarkdown', 'knitr', 'caret', 'randomForest', 'xgboost', 'data.table', 'IRkernel'), repos='https://cloud.r-project.org')"'''
    run_command(r_packages_cmd, check=False)
    
    # Install R kernel for Jupyter
    run_command('R -e "IRkernel::installspec(user = FALSE)"', check=False)
    
    print_success("R environment installed")


def install_database_tools(distro: Distro) -> None:
    """Install database management and query tools"""
    print_header("Installing Database Tools")
    
    if distro == Distro.UBUNTU_DEBIAN:
        run_command("apt install -y postgresql postgresql-contrib libpq-dev", check=False)
        run_command("apt install -y mysql-server libmysqlclient-dev", check=False)
        run_command("apt install -y mongodb-org", check=False)
        
    elif distro == Distro.FEDORA:
        run_command("dnf install -y postgresql-server postgresql-contrib", check=False)
        run_command("dnf install -y mysql-server mysql-devel", check=False)
        run_command("dnf install -y mongodb-org", check=False)
        
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm postgresql", check=False)
        run_command("pacman -S --noconfirm mariadb", check=False)
        run_command("pacman -S --noconfirm mongodb", check=False)
    
    # Install DBeaver (universal database tool)
    if distro == Distro.UBUNTU_DEBIAN:
        run_command("wget -O /tmp/dbeaver.deb https://dbeaver.io/files/dbeaver-ce_latest_amd64.deb", check=False)
        run_command("dpkg -i /tmp/dbeaver.deb || apt-get -f install -y", check=False)
        run_command("rm /tmp/dbeaver.deb", check=False)
    elif distro == Distro.FEDORA:
        run_command("wget -O /tmp/dbeaver.rpm https://dbeaver.io/files/dbeaver-ce-latest-stable.x86_64.rpm", check=False)
        run_command("dnf install -y /tmp/dbeaver.rpm", check=False)
        run_command("rm /tmp/dbeaver.rpm", check=False)
    elif distro == Distro.ARCH:
        run_command("pacman -S --noconfirm dbeaver", check=False)
    
    print_success("Database tools installed")


def install_data_engineering_tools() -> None:
    """Install data engineering and ETL tools"""
    print_header("Installing Data Engineering Tools")
    
    de_packages = [
        "apache-airflow",
        "pyspark",
        "delta-spark",
        "kafka-python",
        "boto3",
        "google-cloud-storage",
        "azure-storage-blob",
        "minio",
        "redis",
        "celery",
    ]
    
    packages_str = " ".join(de_packages)
    run_command(f"pip3 install --user {packages_str}", check=False)
    
    print_success("Data engineering tools installed")


def install_nlp_tools() -> None:
    """Install NLP libraries and tools"""
    print_header("Installing NLP Tools")
    
    nlp_packages = [
        "spacy",
        "nltk",
        "gensim",
        "textblob",
        "sentence-transformers",
        "langchain",
        "openai",
        "tiktoken",
    ]
    
    packages_str = " ".join(nlp_packages)
    run_command(f"pip3 install --user {packages_str}", check=False)
    
    # Download spaCy English model
    run_command("python3 -m spacy download en_core_web_sm", check=False)
    
    # Download NLTK data
    run_command("python3 -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\"", check=False)
    
    print_success("NLP tools installed")


def install_computer_vision_tools() -> None:
    """Install computer vision libraries"""
    print_header("Installing Computer Vision Tools")
    
    cv_packages = [
        "opencv-python",
        "opencv-contrib-python",
        "pillow",
        "scikit-image",
        "albumentations",
        "ultralytics",  # YOLOv8
        "detectron2",
    ]
    
    packages_str = " ".join(cv_packages)
    run_command(f"pip3 install --user {packages_str}", check=False)
    
    print_success("Computer vision tools installed")


def configure_jupyter_settings() -> None:
    """Configure Jupyter with optimal settings for data science"""
    print_header("Configuring Jupyter Settings")
    
    user = os.environ.get("SUDO_USER", os.environ.get("USER"))
    home = Path.home()
    
    if user:
        import pwd
        try:
            user_info = pwd.getpwnam(user)
            home = Path(user_info.pw_dir)
        except KeyError:
            pass
    
    jupyter_config_dir = home / ".jupyter"
    jupyter_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Jupyter Lab settings
    jupyter_config = '''# Jupyter Lab Configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = True
c.ServerApp.port = 8888
c.ServerApp.token = ''
c.ServerApp.password = ''

# Notebook settings
c.NotebookApp.notebook_dir = '~'

# Memory settings for large datasets
c.NotebookApp.iopub_data_rate_limit = 10000000000
c.NotebookApp.iopub_msg_rate_limit = 10000000000
'''
    
    config_file = jupyter_config_dir / "jupyter_lab_config.py"
    with open(config_file, "w") as f:
        f.write(jupyter_config)
    
    # Set proper ownership
    if user:
        try:
            user_info = pwd.getpwnam(user)
            os.chown(jupyter_config_dir, user_info.pw_uid, user_info.pw_gid)
            os.chown(config_file, user_info.pw_uid, user_info.pw_gid)
        except (KeyError, PermissionError):
            pass
    
    print_success("Jupyter configured")


def install_anaconda() -> None:
    """Install Miniconda for environment management"""
    print_header("Installing Miniconda")
    
    user = os.environ.get("SUDO_USER", os.environ.get("USER"))
    home = Path.home()
    
    if user:
        import pwd
        try:
            user_info = pwd.getpwnam(user)
            home = Path(user_info.pw_dir)
        except KeyError:
            pass
    
    miniconda_path = home / "miniconda3"
    
    if miniconda_path.exists():
        print_warning("Miniconda already installed, skipping...")
        return
    
    # Download and install Miniconda
    run_command("wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh", check=False)
    run_command(f"bash /tmp/miniconda.sh -b -p {miniconda_path}", check=False)
    run_command("rm /tmp/miniconda.sh", check=False)
    
    # Set proper ownership
    if user:
        import pwd
        try:
            user_info = pwd.getpwnam(user)
            run_command(f"chown -R {user_info.pw_uid}:{user_info.pw_gid} {miniconda_path}", check=False)
        except (KeyError, PermissionError):
            pass
    
    # Add conda to shell
    shell_configs = [
        home / ".bashrc",
        home / ".zshrc",
        home / ".config" / "fish" / "config.fish"
    ]
    
    conda_init_bash = f'''
# >>> conda initialize >>>
eval "$({miniconda_path}/bin/conda shell.bash hook)"
# <<< conda initialize <<<
'''
    
    conda_init_fish = f'''
# >>> conda initialize >>>
eval {miniconda_path}/bin/conda "shell.fish" "hook" | source
# <<< conda initialize <<<
'''
    
    for config in shell_configs:
        if config.exists():
            with open(config, "a") as f:
                if "fish" in str(config):
                    f.write(conda_init_fish)
                else:
                    f.write(conda_init_bash)
    
    print_success("Miniconda installed")
    print_step("Restart your terminal or run 'source ~/.bashrc' to use conda")


def setup_git_config() -> None:
    """Setup basic Git configuration"""
    print_header("Setting up Git Configuration")
    
    user = os.environ.get("SUDO_USER", os.environ.get("USER"))
    home = Path.home()
    
    if user:
        # Get actual user home directory
        import pwd
        try:
            user_info = pwd.getpwnam(user)
            home = Path(user_info.pw_dir)
        except KeyError:
            pass
    
    gitconfig = """[init]
    defaultBranch = main
[color]
    ui = auto
[core]
    editor = vim
    autocrlf = input
[pull]
    rebase = false
[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    lg = log --oneline --graph --decorate
"""
    
    gitconfig_file = home / ".gitconfig"
    
    # Only write if file doesn't exist
    if not gitconfig_file.exists():
        with open(gitconfig_file, "w") as f:
            f.write(gitconfig)
        
        if user:
            import pwd
            try:
                user_info = pwd.getpwnam(user)
                os.chown(gitconfig_file, user_info.pw_uid, user_info.pw_gid)
            except (KeyError, PermissionError):
                pass
    
    print_success("Git configuration created")


def create_summary_report() -> None:
    """Create a summary of what was installed"""
    print_header("Installation Summary")
    
    summary = """
╔══════════════════════════════════════════════════════════════════════════╗
║                       INSTALLATION COMPLETE!                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Core Development Tools:                                                 ║
║  ✓ Git, Vim, Neovim, tmux, zsh, fish                                    ║
║  ✓ Python 3 with pip, venv, poetry, and dev tools                       ║
║  ✓ Docker and Docker Compose                                            ║
║  ✓ Visual Studio Code                                                   ║
║  ✓ Miniconda for environment management                                 ║
║                                                                          ║
║  Applications:                                                           ║
║  ✓ Discord (chat & voice)                                               ║
║  ✓ DBeaver (Universal DB client)                                        ║
║                                                                          ║
║  Data Science & Analytics Stack:                                         ║
║  ✓ NumPy, Pandas, Polars, SciPy                                         ║
║  ✓ Matplotlib, Seaborn, Plotly, Altair                                  ║
║  ✓ Scikit-learn, XGBoost, LightGBM, CatBoost                            ║
║  ✓ Jupyter Lab & Notebook with extensions                               ║
║  ✓ Streamlit, Dash, Gradio for apps                                     ║
║                                                                          ║
║  Machine Learning & Deep Learning:                                       ║
║  ✓ TensorFlow (GPU/CPU auto-detected)                                   ║
║  ✓ PyTorch (GPU/CPU auto-detected)                                      ║
║  ✓ Transformers, Keras, ONNX                                            ║
║  ✓ SpaCy, NLTK, LangChain (NLP)                                         ║
║  ✓ OpenCV, scikit-image (Computer Vision)                               ║
║                                                                          ║
║  MLOps & Data Engineering:                                               ║
║  ✓ MLflow, Weights & Biases, DVC                                        ║
║  ✓ Apache Airflow, Prefect, Dagster                                     ║
║  ✓ PySpark, DuckDB, Apache Kafka                                        ║
║                                                                          ║
║  Databases:                                                              ║
║  ✓ PostgreSQL, MySQL, MongoDB                                           ║
║  ✓ SQLAlchemy, psycopg2, pymysql                                        ║
║                                                                          ║
║  R Environment:                                                          ║
║  ✓ R with tidyverse, ggplot2, caret                                     ║
║  ✓ R kernel for Jupyter                                                 ║
║                                                                          ║
║  Gaming Components:                                                      ║
║  ✓ NVIDIA GPU drivers (if applicable)                                   ║
║  ✓ Steam with Vulkan support                                            ║
║  ✓ Proton-GE, GameMode, MangoHud                                        ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  NEXT STEPS:                                                             ║
║                                                                          ║
║  1. REBOOT your system to load NVIDIA drivers                           ║
║                                                                          ║
║  2. Restart terminal or run: source ~/.bashrc                           ║
║     This activates Conda and updates PATH                                ║
║                                                                          ║
║  3. Configure Git:                                                       ║
║     git config --global user.name "Your Name"                           ║
║     git config --global user.email "your@email.com"                     ║
║                                                                          ║
║  4. Start Jupyter Lab:                                                   ║
║     jupyter lab                                                          ║
║                                                                          ║
║  5. Create a conda environment for your project:                         ║
║     conda create -n myproject python=3.11                                ║
║     conda activate myproject                                             ║
║                                                                          ║
║  6. Test GPU access:                                                     ║
║     python -c "import torch; print(torch.cuda.is_available())"          ║
║     python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" ║
║                                                                          ║
║  7. For Steam gaming: Enable Proton-GE in Settings > Compatibility      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(summary)


def main():
    """Main entry point"""
    print_header("Linux Data Science & Gaming Setup Script")
    
    # Check for root privileges
    if not check_root():
        print_error("This script must be run as root (use sudo)")
        sys.exit(1)
    
    # Detect distribution
    distro = detect_distro()
    if distro == Distro.UNKNOWN:
        print_error("Unsupported distribution. This script supports Ubuntu/Debian, Fedora, and Arch Linux.")
        sys.exit(1)
    
    try:
        # System update
        update_system(distro)
        
        # Core development environment
        install_essential_packages(distro)
        install_python_dev(distro)
        install_docker(distro)
        install_vscode(distro)
        install_discord(distro)
        install_extra_dev_tools(distro)
        setup_git_config()
        
        # Data Science & Analytics Stack
        install_anaconda()
        install_data_science_stack()
        install_jupyter_ecosystem()
        configure_jupyter_settings()
        install_deep_learning_frameworks()
        install_data_visualization_tools(distro)
        install_mlops_tools()
        install_r_environment(distro)
        install_database_tools(distro)
        install_data_engineering_tools()
        install_nlp_tools()
        install_computer_vision_tools()
        
        # Optional: Node.js for web dashboards
        install_nodejs(distro)
        
        # Gaming setup
        install_nvidia_drivers(distro)
        configure_nvidia_for_gaming()
        install_steam(distro)
        install_proton_ge()
        install_gamemode(distro)
        install_mangohud(distro)
        configure_gaming_tweaks()
        
        # Summary
        create_summary_report()
        
    except KeyboardInterrupt:
        print_warning("\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
