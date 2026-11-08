from pylsl import resolve_stream

def verify_lsl_stream():
    print("Resolving LSL EEG stream ...")
    # Try to resolve an EEG stream (this checks all available streams of type 'EEG')
    streams = resolve_stream('type', 'EEG', timeout=5)

    # If streams are found, print the details
    if streams:
        print(f"Found {len(streams)} LSL EEG stream(s).")
        for stream in streams:
            print(f"Stream Name: {stream.name()}")
            print(f"Stream Type: {stream.type()}")
            print(f"Channel Count: {stream.channel_count()}")
            print(f"Nominal Sampling Rate: {stream.nominal_srate()} Hz")
            print("Stream is ready and streaming data!")
    else:
        print("No LSL EEG streams found. Ensure the RDA → LSL bridge is running.")

if __name__ == "__main__":
    verify_lsl_stream()
