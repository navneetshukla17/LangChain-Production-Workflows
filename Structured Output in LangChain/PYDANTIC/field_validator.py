from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    cgpa: Optional[float] = Field(gt=6, lt=11, default=7, description='cpga must be greater than 6 and less than 11')


new_student2 = {
    'email': 'abc@gmail.com',
    'cgpa': 7
}

new_student3 = {
    'cgpa': 7
}

new_student = {
    'name': 'Tushar',
    'email': 'abc@yahoo.com',
    'cgpa': 8
}

student = Student(**new_student3)
print(student)

