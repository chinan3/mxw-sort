import spikeinterface.extractors as se

# Reads maxwell recording from an H5 file, returns SpikeInterface recording object
def read_maxwell(h5_path: str, stream_name: str):

    return se.MaxwellRecordingExtractor(h5_path, stream_name=stream_name)

# Auto-detects available wells in a Maxwell H5 file, returns a tuple of well indices
def get_available_wells(h5_path: str) -> tuple[int, ...]:

    # MaxwellRecordingExtractor can list available streams
    try:
        # Get all stream names from the file
        extractor = se.MaxwellRecordingExtractor
        stream_names = extractor.get_stream_names(h5_path)

        # Get well indices from common names like "well000" or "well001"
        wells = []
        for stream in stream_names:
            if stream.startswith("well"):
                try:
                    well_idx = int(stream[4:])
                    wells.append(well_idx)
                except ValueError:
                    continue

        return tuple(sorted(wells)) if wells else (0, 1, 2, 3, 4, 5)
    except Exception: # Could allow silent errors as implimented. FUTURE: Add detailed logs and specific exceptions
        # Default to 6 wells if detection fails
        return (0, 1, 2, 3, 4, 5)
