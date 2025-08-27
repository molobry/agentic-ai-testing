#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentic-ai-testing",
    version="1.0.0",
    author="Michal",
    author_email="your-email@example.com",
    description="An intelligent BDD-driven testing framework that uses AI to analyze web pages and automatically execute test scenarios",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/molobry/agentic-ai-testing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Quality Assurance",
    ],
    python_requires=">=3.9",
    install_requires=[
        "playwright>=1.40.0",
        "beautifulsoup4>=4.12.2",
        "lxml>=4.9.3",
        "openai>=1.3.0",
        "anthropic>=0.34.0",
        "python-dotenv>=1.0.0",
        "Pillow>=10.1.0",
        "opencv-python>=4.8.1.78",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentic-test=run_test:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.xml", "*.json", "*.md"],
    },
)