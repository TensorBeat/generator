import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
import asyncio
from grpclib.server import Server
from grpclib.reflection.service import ServerReflection
from tensorbeat.sarosh_gen import SaroshGeneratorBase, GenerateMusicResponse


class GeneratorService(SaroshGeneratorBase):
    async def generate_music(self, notes) -> "GenerateMusicResponse":
        handler = InputHandler()
        print("Received notes " + str(notes))
        song = handler.generateSong(notes)
        response = GenerateMusicResponse(song)
        return response


async def start_server():
    host = "127.0.0.1"
    port = 3491
    services = [GeneratorService()]
    server = Server(services)
    await server.start(host, port)
    print("Listening on host {host} on port {port}".format(host=host, port=port))
    await server.wait_closed()


class InputHandler:
    def __init__(self):
        pass

    def generateSong(self, seed):
        g = Generator(n_ts=seed)
        if isinstance(seed, str):
            pass
            notes = g.generate()
        else:
            notes = g.generate()
            return notes
        return 0


class Generator:

    def __init__(self, n_ts=[], mod='weights.hdf5'):
        self.model = mod
        self.notes_cond = n_ts

    def load_notes(self, n, fromweb=False):
        if fromweb == False:
            self.notes_cond = n

    def consolidate_notes(self, a, b):
        d = len(set(a))
        la = len(a)
        lb = len(b)
        final_notes = []
        if lb < la:
            sl = [a[i] for i in range(len(a)) if a[i] not in b]
            final_notes = b
            for i in range(len(sl)):
                final_notes.append(sl[i])
                if len(set(final_notes)) == d:
                    break
        else:
            final_notes = a[0:len(b)]
        return final_notes

    def generate(self):
        with open('data/notes', 'rb') as filepath:
            pre_notes = pickle.load(filepath)

        if len(self.notes_cond) == 0:
            post_notes = self.get_notes()
        else:
            post_notes = self.notes_cond

        notes = self.consolidate_notes(pre_notes, post_notes)

        pitchnames = sorted(set(item for item in notes))
        n_vocab = len(set(notes))

        network_input, normalized_input = self.prepare_sequences(notes, pitchnames, n_vocab)
        cat_net_in, net_out = self.prepare_sequences_categorical(notes, n_vocab)
        model = self.create_network(normalized_input, n_vocab)
        print("Loading weights")
        self.train(model, cat_net_in, net_out, 0)
        print("Generating songs")
        prediction_output = self.generate_notes(model, network_input, pitchnames, n_vocab)
        print("Creating midi")
        song_notes = self.create_midi(prediction_output)
        return list(map(lambda x: x.name, song_notes))

    def prepare_sequences(self, notes, pitchnames, n_vocab):
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        sequence_length = 100
        network_input = []
        output = []
        for i in range(0, len(notes) - sequence_length):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)
        normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        normalized_input = normalized_input / float(n_vocab)

        return (network_input, normalized_input)

    def prepare_sequences_categorical(self, notes, n_vocab):
        sequence_length = 100

        pitchnames = sorted(set(item for item in notes))

        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        network_input = []
        network_output = []

        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

        network_input = network_input / float(n_vocab)

        network_output = utils.to_categorical(network_output)

        return (network_input, network_output)

    def get_notes(self):
        """ Get all the notes and chords from the midi files in the ./midi_songs directory """
        notes = []

        for file in glob.glob("midi_songs/*.mid"):
            midi = converter.parse(file)

            print("Parsing %s" % file)

            notes_to_parse = None

            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        return notes

    def create_network(self, network_input, n_vocab):
        """ create the structure of the neural network """
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            recurrent_dropout=0.3,
            return_sequences=True
        ))
        model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
        model.add(LSTM(512))
        model.add(BatchNorm())
        model.add(Dropout(0.3))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNorm())
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.load_weights(self.model)

        return model

    def train(self, model, network_input, network_output, eps=200):
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        model.fit(network_input, network_output, epochs=eps, batch_size=128, callbacks=callbacks_list)

    def generate_notes(self, model, network_input, pitchnames, n_vocab):
        start = numpy.random.randint(0, len(network_input) - 1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

        pattern = network_input[start]
        prediction_output = []
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        return prediction_output

    def create_midi(self, prediction_output, outputs=[]):
        offset = 0
        output_notes = outputs
        for pattern in prediction_output:
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            offset += 0.5

        midi_stream = stream.Stream(output_notes)

        # midi_stream.write('midi', fp='test_output.mid')
        return output_notes


if __name__ == '__main__':
    asyncio.run(start_server())
