from pydantic import BaseModel

class Student(BaseModel):
    name: str = 'Tushar' # setting default value

new_student = {}

student = Student(**new_student)
print(student.name)
