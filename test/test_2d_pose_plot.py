from turtle import Screen, Turtle, mainloop

# Move car
def turn_right():
    car.right(20)

def turn_left():
    car.left(20)

def move():
    car.forward(1)
    screen.update()
    screen.ontimer(move, 25)

# Screen
screen = Screen()
screen.title('Car')
screen.bgcolor('black')
screen.setup(width=1200, height=1200)
screen.tracer(0)

# Track
track = Turtle()
track.hideturtle()
track.color('white')

track.penup()
track.goto(-550, 0)
track.pendown()

track.goto(-550, 300)
track.goto(-100, 370)
track.goto(100, 210)
track.goto(300, 380)
track.goto(580, 100)
track.goto(570, -300)
track.goto(300, -370)
track.goto(0, -250)
track.goto(-300, -200)
track.goto(-570, -250)
track.goto(-550, 0)

track.penup()
track.goto(-450, 0)
track.pendown()

track.goto(-450, 230)
track.goto(-150, 250)
track.goto(100, 100)
track.goto(300, 200)
track.goto(460, 100)
track.goto(450, -220)
track.goto(300, -250)
track.goto(0, -130)
track.goto(-300, -100)
track.goto(-450, 0)

# Start line
line = Turtle()
line.hideturtle()
line.color('white')

line.penup()
line.setx(-550)
line.pendown()
line.setx(-450)