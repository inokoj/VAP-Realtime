import socket
import argparse
import math
import sys
import rvap.common.util as util

def draw_bar(value, length=30):
    bar_len = min(length, max(0, int(abs(value) * length)))
    bar = '█' * bar_len
    rest = '-' * (length - bar_len)
    return bar + rest

def rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(x * x for x in values) / len(values))

def safe_first(vec, default=0.0):
    return vec[0] if isinstance(vec, (list, tuple)) and len(vec) > 0 else default

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
                vap_result = util.conv_bytearray_2_vapresult_bc(data)
                x1_rms = rms(vap_result.get('x1', []))
                x2_rms = rms(vap_result.get('x2', []))

                p_bc_react = safe_first(vap_result.get('p_bc_react', []))
                p_bc_emo = safe_first(vap_result.get('p_bc_emo', []))

                sys.stdout.write("\x1b[H")
                print(f"t: {vap_result.get('t', 0):.3f}")
                print(f"x1 [mic]:    {draw_bar(x1_rms)} ({x1_rms:.4f})")
                print(f"x2 [ref]:    {draw_bar(x2_rms)} ({x2_rms:.4f})")
                print(f"p_bc_react:  {draw_bar(p_bc_react)} ({p_bc_react:.4f})")
                print(f"p_bc_emo:    {draw_bar(p_bc_emo)} ({p_bc_emo:.4f})")
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