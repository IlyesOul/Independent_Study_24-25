print("Please think of a number between 0 and 100!")

high = 100
low = 0
guess = 50

print(f"Is your secret number {guess}?")
print(f"Enter 'h' to indicate the guess is too high. Enter 'l' to indicate the guess is too low.")
response = input("Enter 'c' to indicate I guessed correctly. ")
while(response != 'c'):
    while response != 'h' and response != 'l' and response != 'c':
        response = input("I don't understand your input, please enter again")
    if(response == 'h'):
        high = guess
    elif(response == 'l'):
        low = guess
    guess = int((high+low)/2)
    print(f"Is your secret number {guess}?")
    print(f"Enter 'h' to indicate the guess is too high. Enter 'l' to indicate the guess is too low.")
    response = input("Enter 'c' to indicate I guessed correctly. ")
print(f"Game over. Your secret number was: {guess}")