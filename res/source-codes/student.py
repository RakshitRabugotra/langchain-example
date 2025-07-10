class Student:
    def __init__(self, name, student_id, major):
        self.name = name
        self.student_id = student_id
        self.major = major
        self.courses = []  # Initialize an empty list for courses

    def enroll_course(self, course_name):
        self.courses.append(course_name)
        print(f"{self.name} enrolled in {course_name}.")

    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Student ID: {self.student_id}")
        print(f"Major: {self.major}")
        if self.courses:
            print(f"Enrolled Courses: {', '.join(self.courses)}")
        else:
            print("No courses enrolled.")

# Creating Student objects
student1 = Student("Alice Smith", "S001", "Computer Science")
student2 = Student("Bob Johnson", "S002", "Engineering")

# Using methods
student1.enroll_course("Introduction to Programming")
student2.enroll_course("Calculus I")
student1.enroll_course("Data Structures")

# Displaying information
student1.display_info()
print("-" * 20)
student2.display_info()