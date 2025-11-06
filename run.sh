#!/bin/bash

# Nong-View2 Pipeline Runner for Linux/Mac

echo "==============================================="
echo "Nong-View2 Agricultural AI Analysis Pipeline"
echo "==============================================="
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check command line arguments
if [ $# -eq 0 ]; then
    echo
    echo "Usage: ./run.sh [command] [options]"
    echo
    echo "Commands:"
    echo "  full      - Run full pipeline"
    echo "  test      - Run tests"
    echo "  example   - Run example script"
    echo "  help      - Show help"
    echo
    exit 0
fi

# Execute commands
case "$1" in
    full)
        echo "Running full pipeline..."
        shift
        python main.py "$@"
        ;;
    test)
        echo "Running tests..."
        if command -v pytest &> /dev/null; then
            pytest tests/ -v
        else
            python tests/test_pipeline.py
        fi
        ;;
    example)
        echo "Running example..."
        python run_example.py
        ;;
    help)
        python main.py --help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './run.sh' without arguments for help"
        exit 1
        ;;
esac

echo
echo "==============================================="
echo "Process completed"
echo "==============================================="