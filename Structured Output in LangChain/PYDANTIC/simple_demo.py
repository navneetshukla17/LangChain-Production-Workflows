from pydantic import BaseModel

class Student(BaseModel):
    name: str

new_student = {'name': '111'}

student = Student(**new_student)

print(student)
