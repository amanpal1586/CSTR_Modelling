"""End-to-end: generate data -> train surrogate -> evaluate & plot."""
from src import generate_data, train_surrogate, evaluate

if __name__ == "__main__":
    generate_data.main()
    train_surrogate.main()
    evaluate.main()
