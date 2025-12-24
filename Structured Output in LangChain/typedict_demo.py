from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    salary: int


# First person
new_person = Person({
    'name': 'Tushar',
    'age': 22,
    'salary': 50000
})

# Second person
another_person = Person({
    'name': 'Hardik',
    'age': 23,
    'salary': 100000
})

print(another_person)

