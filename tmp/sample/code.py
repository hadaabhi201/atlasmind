# Line 1
def fibonacci(n):
    # Line 3
    if n <= 1:
        return n
    # Line 6
    return fibonacci(n - 1) + fibonacci(n - 2)

# Line 8
def main():
    # Line 10
    result = fibonacci(6)
    print("Fibonacci result:", result)

# Line 13
if __name__ == "__main__":
    main()
