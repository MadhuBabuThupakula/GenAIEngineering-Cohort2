import sys
print("hello world  ")
a= 5
b = 10
def add(a, b):
    """Add two numbers and return the result."""
    return a + b

#my_add = add(a, b)
#print(f"{a} + {b} = {my_add}")

def subtract(a, b):
    """Subtract b from a and return the result."""
    return a - b
#my_subtract = subtract(a, b)
#print(f"{a} - {b} = {my_subtract}")

def multiply(a, b):
    """Multiply two numbers and return the result."""
    return a * b
#my_multiply = multiply(a, b)
#print(f"{a} * {b} = {my_multiply}")      

if __name__ == "__main__":  
    choice = sys.argv[1]
    num1 = float(sys.argv[2])
    num2 = float(sys.argv[3])

    if choice.upper() == 'ADD':
        result = add(num1, num2)
        print(f"{num1} + {num2} = {result}")
    elif choice.upper() == 'SUB':
        result = subtract(num1, num2)
        print(f"{num1} - {num2} = {result}")
    elif choice.upper() == 'MUL':
        result = multiply(num1, num2)
        print(f"{num1} * {num2} = {result}")
    else:
        print("Invalid input")