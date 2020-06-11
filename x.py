class Animal():
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def say(self, sound):
        print(f"ANIMAL CLASS {self.name} says {sound}")


class Cat(Animal):
    def __init__(self, name, species, breed):
        super().__init__(name, species)
        self.breed = breed

    def sound(self):
        return self.say("MEOW")

blue = Cat("blue", "cat", "Persian")

print(isinstance(blue, Animal))
print(isinstance(blue, Cat))

print(blue.sound())
    