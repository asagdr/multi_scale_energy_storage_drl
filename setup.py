"""
多尺度储能系统深度强化学习项目安装脚本
Multi-scale Energy Storage DRL System Setup Script
"""

from setuptools import setup, find_packages
import os
import sys

# 读取README文件
def read_readme():
    """读取README文件内容"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "多尺度储能系统深度强化学习项目"

# 读取requirements文件
def read_requirements():
    """读取requirements.txt文件"""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

# 项目版本
VERSION = "0.1.0"

# 项目元数据
AUTHOR = "DRL Energy Storage Team"
AUTHOR_EMAIL = "asagdr@example.com"
DESCRIPTION = "A multi-scale deep reinforcement learning system for energy storage optimization"
URL = "https://github.com/asagdr/multi-scale-energy-storage-drl"

# Python版本要求
PYTHON_REQUIRES = ">=3.8"

# 分类信息
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

# 关键词
KEYWORDS = [
    "deep reinforcement learning",
    "energy storage",
    "battery management",
    "hierarchical control",
    "multi-scale optimization",
    "smart grid",
    "renewable energy"
]

# 项目URLs
PROJECT_URLS = {
    "Documentation": f"{URL}/docs",
    "Source": URL,
    "Tracker": f"{URL}/issues",
}

# 控制台脚本
CONSOLE_SCRIPTS = [
    "drl-energy-storage=main:main",
    "drl-train=scripts.train:main",
    "drl-evaluate=scripts.evaluate:main",
    "drl-experiment=scripts.run_experiment:main",
]

# 额外依赖
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
        "sphinx>=4.1.0",
        "sphinx-rtd-theme>=0.5.0",
    ],
    "gpu": [
        "torch>=1.9.0+cu111",
        "torchvision>=0.10.0+cu111",
    ],
    "visualization": [
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "tensorboard>=2.7.0",
        "wandb>=0.12.0",
    ],
    "analysis": [
        "SALib>=1.4.0",
        "statsmodels>=0.12.0",
        "scipy>=1.7.0",
    ]
}

# 包数据
PACKAGE_DATA = {
    "config": ["*.yaml", "*.json"],
    "data": ["*.csv", "*.json"],
    "models": ["*.pth", "*.pt"],
    "experiments": ["*.json", "*.yaml"],
}

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        sys.exit("Python 3.8 or higher is required. You are using Python {}.{}.{}".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro))

def main():
    """主安装函数"""
    check_python_version()
    
    setup(
        name="multi-scale-energy-storage-drl",
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url=URL,
        project_urls=PROJECT_URLS,
        packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
        classifiers=CLASSIFIERS,
        python_requires=PYTHON_REQUIRES,
        install_requires=read_requirements(),
        extras_require=EXTRAS_REQUIRE,
        package_data=PACKAGE_DATA,
        include_package_data=True,
        entry_points={
            "console_scripts": CONSOLE_SCRIPTS,
        },
        keywords=" ".join(KEYWORDS),
        zip_safe=False,
        
        # 测试配置
        test_suite="tests",
        tests_require=EXTRAS_REQUIRE["dev"],
        
        # 数据文件
        data_files=[
            ("config", ["config/default_config.yaml"]),
            ("docs", ["README.md", "LICENSE"]),
        ],
    )

if __name__ == "__main__":
    main()
