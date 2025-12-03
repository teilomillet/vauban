from contextlib import nullcontext, contextmanager

@contextmanager
def faulty_generator():
    yield from nullcontext()

if __name__ == "__main__":
    try:
        with faulty_generator():
            print("Inside context")
    except TypeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")
