

def fizzbuzz():
    size = 100
    for i in range(size):
        res =""
        if i % 3 == 0:res += "Fizz"
        if i % 5 == 0:res += "Buzz"
        print(i,res)



if __name__ == "__main__":
    fizzbuzz()
# end main

    