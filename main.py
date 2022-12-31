from os import system, name
import smplscript

def cs():
    _ = system("cls") if name == 'nt' else system('clear')

if __name__ == "__main__":
    cs()
    while True:
        inp = input('smpl > ')
        if inp == 'cls':
            cs()
        elif inp == 'stop':
            break
        else:
            res, error = smplscript.run('stdin', inp)

            if error: print(error.as_string())
            elif res: print(res)