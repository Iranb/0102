from peewee import SqliteDatabase, Model
from peewee import CharField, DateField

db = SqliteDatabase('vehicle.db')

class Vehicle(Model):
    licence = CharField()
