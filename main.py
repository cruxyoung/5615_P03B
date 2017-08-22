from data_examining.data_fetch import fetch_data
import os


# customer = fetch_data('source_datasets/Supermarket_customer.csv')

# print(customer.values)


if __name__ == '__main__':
    # looping to wait for the commands from customer
    while(1):
        raw_cmd = input("Enter command: ")
        cmd = raw_cmd.split()
        if not cmd:
            print("You need to enter a command...")
            continue
        if cmd[0].lower() == 'fetch':
            customer = fetch_data('source_datasets/{}'.format(cmd[1]))
            # print(customer.values)
        elif cmd[0].lower() == 'list':
            files = os.listdir('source_datasets')
            for f in files:
                print(f)

        elif cmd[0].lower() == 'print':
            print(customer.values)