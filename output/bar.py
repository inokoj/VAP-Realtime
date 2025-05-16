import socket
import argparse
import math
import sys
import rvap.common.util as util


def draw_bar(value, length=30):
    bar_len = min(length, max(0, int(value * length)))
    return '█' * bar_len + '-' * (length - bar_len)


def draw_symmetric_bar(value, length=30):
    max_len = length // 2
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        pos = int(value * max_len)
        return ' ' * max_len + '│' + '█' * pos + ' ' * (max_len - pos)
    else:
        neg = int(-value * max_len)
        return ' ' * (max_len - neg) + '█' * neg + '│' + ' ' * max_len


def draw_balance_bar(value, length=30):
    """
    0.5を中心(│)として、それより小さいと左にバー、大きいと右にバーを描画
    """
    max_len = length // 2
    value = max(0.0, min(1.0, value))
    diff = value - 0.5
    if diff > 0:
        right = int(diff * 2 * max_len + 0.5)  # 0.5~1.0: 0~max_len
        return ' ' * max_len + '│' + '█' * right + ' ' * (max_len - right)
    else:
        left = int(-diff * 2 * max_len + 0.5)  # 0.0~0.5: max_len~0
        return ' ' * (max_len - left) + '█' * left + '│' + ' ' * max_len


def rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(x * x for x in values) / len(values))


def print_inline(label, value1, value2):
    line = f"{label}: {draw_symmetric_bar(value1)} ({value1: .3f})  {draw_symmetric_bar(value2)} ({value2: .3f})"
    sys.stdout.write("\r" + line)
    sys.stdout.flush()


def process_client(server_ip='127.0.0.1', port_num=50008):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock_server:
        sock_server.connect((server_ip, port_num))

        print("\x1b[2J")  # 初期クリア

        try:
            while True:
                data_size = sock_server.recv(4)
                if not data_size:
                    break
                size = int.from_bytes(data_size, 'little')

                data = b''
                while len(data) < size:
                    chunk = sock_server.recv(size - len(data))
                    if not chunk:
                        raise ConnectionError("Socket closed")
                    data += chunk

                vap_result = util.conv_bytearray_2_vapresult(data)

                x1_rms = rms(vap_result['x1'])
                x2_rms = rms(vap_result['x2'])
                p_now = vap_result['p_now']
                p_future = vap_result['p_future']

                sys.stdout.write("\x1b[H")  # カーソルだけ戻す
                print(f"t: {vap_result['t']:.3f}")
                print(f"x1 [mic]:    {draw_bar(x1_rms)} ({x1_rms:.4f})")
                print(f"x2 [ref]:    {draw_bar(x2_rms)} ({x2_rms:.4f})")
                print(f"p_now:       {draw_balance_bar(p_now[0])} ({p_now[0]: .3f})")
                print(f"p_future:    {draw_balance_bar(p_future[0])} ({p_future[0]: .3f})")

        except KeyboardInterrupt:
            print("\n🔴 Ctrl+C detected. Exiting gracefully.")
        except Exception as e:
            print("\n🔴 Disconnected from the server due to error:")
            print(e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default='127.0.0.1')
    parser.add_argument("--port_num", type=int, default=50008)
    args = parser.parse_args()

    process_client(args.server_ip, args.port_num)