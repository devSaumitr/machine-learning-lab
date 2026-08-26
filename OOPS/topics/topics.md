# Day 4 — OOP: Classes, Objects & Object-Oriented Design

## Objective

Cross the line from "writing Python scripts" to "designing with Python": model real things as classes, create objects with constructors, protect data with encapsulation, reuse behavior with inheritance, write polymorphic code, hide complexity with abstraction, and control how objects behave in normal Python expressions through magic methods. This matches the CampusX OOP lectures (Part 1 — Class/Object, Part 2 — Encapsulation & Static Keyword) and your existing practice files.

## Topics

1. [Classes & Objects](./01-classes-objects.md) — `class`, `__init__`, `self`, attributes
2. [Instance / Class / Static Methods](./02-instance-class-static-methods.md) — `@classmethod`, `@staticmethod`, `@property`
3. [Encapsulation](./03-encapsulation.md) — private members, name mangling, getters/setters
4. [Inheritance](./04-inheritance.md) — `super()`, method overriding, MRO
5. [Polymorphism](./05-polymorphism.md) — duck typing, overriding, operator overloading
6. [Abstraction](./06-abstraction.md) — `ABC`, `@abstractmethod`
7. [Magic (Dunder) Methods](./07-magic-methods.md) — `__str__`, `__repr__`, `__len__`, `__eq__`, `__add__`
8. [File Handling](./file%20handling/08-file-handling.md) — `open()` modes, reading/writing, `with`, encoding, buffering
9. [Exception & Error Handling](./exception%20handling/09-exception-handling.md) — error types, exception hierarchy, `try/except/else/finally`, `raise`, custom exceptions

## Learning Order

```text
classes & objects        (the core idea: blueprint + instance)
    ↓
methods                  (instance vs class vs static — "static keyword" lecture)
    ↓
encapsulation            (protecting data — Part 2 lecture)
    ↓
inheritance              (reusing and extending classes)
    ↓
polymorphism             (one interface, many behaviors)
    ↓
abstraction              (hiding implementation behind a contract)
    ↓
magic methods            (making objects behave like Python natives)
    ↓
file handling            (reading/writing files — with, modes, encoding)
    ↓
exception handling       (error types, try/except, raise, custom errors)
```

## Prerequisites

- [Day 3](../CORE_PYTHON/topics/Day3/topics.md): functions, `*args`/`**kwargs`, scope & LEGB, modules, `__name__`/`__main__`
- [Day 2](../CORE_PYTHON/topics/Day2/topics.md): mutability & copying, exceptions, `None`
- Day 1 basics: variables, data types, loops, conditions, data structures

## Previously Covered

- Day 1: variables, data types, I/O, operators, conditions, loops, strings
- Day 2: functions, args/kwargs, exceptions, mutability, scope/LEGB, truthy/falsy, `None`
- Day 3: lambda, map/filter/reduce, recursion, iterators, generators, modules/packages, `__name__`/`__main__`
- CampusX lectures watched: *OOP Part 1 — Class/Object*, *OOP Part 2 — Encapsulation & Static Keyword* (`LECTURES/PYTHON/`)
- Practice files that map to these topics: `oops.py` (class/object/`__init__`), `OOPS2.PY` (encapsulation, name mangling), `dunder.py` (magic methods), `inh.py` (`@classmethod`, `@property`, setter), `poly.py` (duck typing, overriding, operator overloading)

## Notes

- `poly.py` has an unfinished `__add__` with a bug (`self.real + other.img`). Topic [05](./05-polymorphism.md) and [07](./07-magic-methods.md) fix that exact code.
- OOP is where ML code starts to become "engineering": sklearn estimators, PyTorch `nn.Module`, data classes — all are classes with methods and magic methods.
- After Day 4: **Python Internals** (memory model, closures, decorators, context managers), then the **NumPy transition**.

## Next

- **Python Internals** — objects & references, memory model, closures, decorators, context managers
- Then **NumPy** (`PYTHON_LIBRARIES/numpy/` already has practice files waiting)
