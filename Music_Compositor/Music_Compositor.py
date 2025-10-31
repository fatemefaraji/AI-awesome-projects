# Enhanced Music Composition System

import os
from typing import TypedDict, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image, Audio
import music21
import tempfile
import random
import json
from datetime import datetime

# Enhanced State Definition
class MusicState(TypedDict):
    musician_input: str
    key: str
    tempo: int
    time_signature: str
    instruments: List[str]
    style: str
    emotion: str
    complexity: str
    melody: str
    harmony: str
    rhythm: str
    structure: str
    composition: str
    midi_file: str
    music_xml: str
    metadata: Dict

# Configuration
class MusicConfig:
    def __init__(self):
        self.available_keys = ['C major', 'C minor', 'G major', 'D major', 'A minor', 'F major']
        self.available_tempos = [60, 80, 100, 120, 140]
        self.available_instruments = ['piano', 'violin', 'cello', 'flute', 'guitar', 'string_quartet']
        self.available_styles = ['classical', 'romantic', 'baroque', 'jazz', 'contemporary', 'minimalist']
        self.available_emotions = ['happy', 'sad', 'mysterious', 'energetic', 'calm', 'dramatic']
        self.complexity_levels = ['simple', 'moderate', 'complex']

# Enhanced LLM Initialization
class MusicLLM:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        chain = ChatPromptTemplate.from_template(prompt) | self.llm
        response = chain.invoke(kwargs)
        return response.content

# Input Parser and Validator
class InputProcessor:
    def __init__(self, config: MusicConfig):
        self.config = config
    
    def parse_input(self, user_input: str) -> Dict:
        """Extract musical parameters from user input"""
        parsed = {
            'key': self._extract_key(user_input),
            'tempo': self._extract_tempo(user_input),
            'style': self._extract_style(user_input),
            'emotion': self._extract_emotion(user_input),
            'instruments': self._extract_instruments(user_input),
            'complexity': self._extract_complexity(user_input)
        }
        return parsed
    
    def _extract_key(self, text: str) -> str:
        text_lower = text.lower()
        for key in self.config.available_keys:
            if key.lower() in text_lower:
                return key
        return random.choice(self.config.available_keys)
    
    def _extract_tempo(self, text: str) -> int:
        # Look for tempo indications
        if 'largo' in text.lower(): return 60
        if 'adagio' in text.lower(): return 70
        if 'andante' in text.lower(): return 90
        if 'moderato' in text.lower(): return 110
        if 'allegro' in text.lower(): return 130
        if 'presto' in text.lower(): return 160
        return random.choice(self.config.available_tempos)
    
    def _extract_style(self, text: str) -> str:
        text_lower = text.lower()
        for style in self.config.available_styles:
            if style in text_lower:
                return style
        return 'classical'
    
    def _extract_emotion(self, text: str) -> str:
        text_lower = text.lower()
        for emotion in self.config.available_emotions:
            if emotion in text_lower:
                return emotion
        return 'calm'
    
    def _extract_instruments(self, text: str) -> List[str]:
        instruments = []
        text_lower = text.lower()
        for instrument in self.config.available_instruments:
            if instrument in text_lower:
                instruments.append(instrument)
        return instruments if instruments else ['piano']
    
    def _extract_complexity(self, text: str) -> str:
        text_lower = text.lower()
        if 'simple' in text_lower or 'easy' in text_lower:
            return 'simple'
        elif 'complex' in text_lower or 'complicated' in text_lower:
            return 'complex'
        return 'moderate'

# Enhanced Musical Node Functions
class MusicNodes:
    def __init__(self, llm: MusicLLM, config: MusicConfig):
        self.llm = llm
        self.config = config
    
    def input_processor(self, state: MusicState) -> Dict:
        """Process and validate user input"""
        processor = InputProcessor(self.config)
        parsed_input = processor.parse_input(state["musician_input"])
        
        return {
            "key": parsed_input["key"],
            "tempo": parsed_input["tempo"],
            "style": parsed_input["style"],
            "emotion": parsed_input["emotion"],
            "instruments": parsed_input["instruments"],
            "complexity": parsed_input["complexity"],
            "time_signature": self._determine_time_signature(parsed_input["style"]),
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "input_parameters": parsed_input,
                "version": "2.0"
            }
        }
    
    def _determine_time_signature(self, style: str) -> str:
        time_signatures = {
            'classical': '4/4',
            'baroque': '4/4',
            'romantic': '3/4',
            'jazz': '4/4',
            'contemporary': '4/4',
            'minimalist': '4/4'
        }
        return time_signatures.get(style, '4/4')
    
    def structure_planner(self, state: MusicState) -> Dict:
        """Plan the musical structure based on style and complexity"""
        prompt = """
        Plan a musical structure for a {style} style composition in {key} with {emotion} emotion.
        Complexity level: {complexity}
        Instruments: {instruments}
        
        Provide a JSON structure with:
        - sections (e.g., introduction, exposition, development, recapitulation, coda)
        - section_lengths in measures
        - harmonic_progression outline
        - dynamic_plan
        - texture_description
        
        Return ONLY valid JSON.
        """
        
        structure_json = self.llm.generate(
            prompt,
            style=state["style"],
            key=state["key"],
            emotion=state["emotion"],
            complexity=state["complexity"],
            instruments=", ".join(state["instruments"])
        )
        
        try:
            structure_data = json.loads(structure_json)
        except:
            # Fallback structure
            structure_data = {
                "sections": ["Introduction", "Main Theme", "Development", "Conclusion"],
                "section_lengths": [4, 8, 8, 4],
                "harmonic_progression": ["I", "IV", "V", "I"],
                "dynamic_plan": ["piano", "crescendo", "forte", "diminuendo"],
                "texture_description": "Homophonic with melodic emphasis"
            }
        
        return {"structure": json.dumps(structure_data)}
    
    def melody_generator(self, state: MusicState) -> Dict:
        """Generate melody based on structure and parameters"""
        structure = json.loads(state["structure"])
        
        prompt = """
        Generate a {emotion} melody in {key} for {style} style.
        Structure: {structure}
        Complexity: {complexity}
        Time Signature: {time_signature}
        
        Provide the melody as a sequence of notes in music21 format with durations.
        Include phrasing and articulation suggestions.
        
        Format: 
        - Note sequence with durations
        - Phrasing marks
        - Dynamic markings
        - Articulation suggestions
        """
        
        melody = self.llm.generate(
            prompt,
            emotion=state["emotion"],
            key=state["key"],
            style=state["style"],
            structure=state["structure"],
            complexity=state["complexity"],
            time_signature=state["time_signature"]
        )
        
        return {"melody": melody}
    
    def harmony_creator(self, state: MusicState) -> Dict:
        """Create harmony that complements the melody"""
        prompt = """
        Create harmonic progression for this melody in {key}:
        Melody: {melody}
        
        Style: {style}
        Emotion: {emotion}
        Structure: {structure}
        
        Provide chords in music21 format that complement the melody.
        Include:
        - Chord symbols
        - Voice leading suggestions
        - Cadence points
        - Harmonic rhythm
        """
        
        harmony = self.llm.generate(
            prompt,
            key=state["key"],
            melody=state["melody"],
            style=state["style"],
            emotion=state["emotion"],
            structure=state["structure"]
        )
        
        return {"harmony": harmony}
    
    def rhythm_designer(self, state: MusicState) -> Dict:
        """Design rhythmic patterns"""
        prompt = """
        Design rhythmic patterns for {style} style composition in {time_signature}.
        Melody: {melody}
        Harmony: {harmony}
        Emotion: {emotion}
        
        Provide:
        - Rhythmic patterns for different sections
        - Syncopation suggestions if appropriate
        - Tempo variations
        - Articulation patterns
        """
        
        rhythm = self.llm.generate(
            prompt,
            style=state["style"],
            time_signature=state["time_signature"],
            melody=state["melody"],
            harmony=state["harmony"],
            emotion=state["emotion"]
        )
        
        return {"rhythm": rhythm}
    
    def style_integrator(self, state: MusicState) -> Dict:
        """Integrate all elements with stylistic consistency"""
        prompt = """
        Integrate these musical elements into a cohesive {style} style composition:
        
        Structure: {structure}
        Melody: {melody}
        Harmony: {harmony}
        Rhythm: {rhythm}
        
        Key: {key}
        Tempo: {tempo}
        Emotion: {emotion}
        Instruments: {instruments}
        
        Provide the complete composition in music21 format with:
        - Proper instrumentation for {instruments}
        - Dynamic markings
        - Articulations
        - Phrasing
        - Expression marks
        """
        
        composition = self.llm.generate(
            prompt,
            style=state["style"],
            structure=state["structure"],
            melody=state["melody"],
            harmony=state["harmony"],
            rhythm=state["rhythm"],
            key=state["key"],
            tempo=state["tempo"],
            emotion=state["emotion"],
            instruments=", ".join(state["instruments"])
        )
        
        return {"composition": composition}

# Enhanced MIDI and MusicXML Converter
class MusicConverter:
    def __init__(self):
        self.instrument_mapping = {
            'piano': music21.instrument.Piano,
            'violin': music21.instrument.Violin,
            'cello': music21.instrument.Cello,
            'flute': music21.instrument.Flute,
            'guitar': music21.instrument.Guitar,
            'string_quartet': music21.instrument.StringInstrument
        }
    
    def create_composition(self, state: MusicState) -> Dict:
        """Convert the composition to MIDI and MusicXML"""
        try:
            # Create a score
            score = music21.stream.Score()
            
            # Parse the composition (this is a simplified approach)
            # In a real implementation, you'd parse the LLM's music21 output
            composition_stream = self._parse_composition(state)
            
            # Add metadata
            metadata = music21.metadata.Metadata()
            metadata.title = f"{state['style'].title()} Composition in {state['key']}"
            metadata.composer = "AI Music Composer"
            metadata.date = datetime.now().strftime("%Y")
            score.insert(0, metadata)
            
            # Add tempo
            tempo = music21.tempo.MetronomeMark(number=state['tempo'])
            score.insert(0, tempo)
            
            # Create parts for each instrument
            for instrument_name in state['instruments']:
                part = self._create_instrument_part(instrument_name, state)
                score.insert(0, part)
            
            # Save MIDI
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_midi:
                score.write('midi', temp_midi.name)
                midi_path = temp_midi.name
            
            # Save MusicXML
            with tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml") as temp_xml:
                score.write('musicxml', temp_xml.name)
                xml_path = temp_xml.name
            
            return {
                "midi_file": midi_path,
                "music_xml": xml_path,
                "composition": "Composition successfully generated"
            }
            
        except Exception as e:
            print(f"Error in composition creation: {e}")
            # Fallback: create a simple composition
            return self._create_fallback_composition(state)
    
    def _parse_composition(self, state: MusicState) -> music21.stream.Stream:
        """Parse the LLM's composition text into music21 objects"""
        # This is a simplified parser - you'd want to enhance this
        # to properly interpret the LLM's music21 format output
        stream = music21.stream.Stream()
        
        # Add some basic notes based on the key
        key = music21.key.Key(state['key'].split()[0])
        scale = key.getScale()
        
        # Create a simple melody using scale notes
        for i in range(16):
            note = music21.note.Note(scale.pitches[i % len(scale.pitches)])
            note.quarterLength = random.choice([0.5, 1.0, 2.0])
            stream.append(note)
        
        return stream
    
    def _create_instrument_part(self, instrument_name: str, state: MusicState) -> music21.stream.Part:
        """Create a part for a specific instrument"""
        part = music21.stream.Part()
        
        # Add instrument
        if instrument_name in self.instrument_mapping:
            instrument = self.instrument_mapping[instrument_name]()
            part.insert(0, instrument)
        
        # Add some notes (simplified)
        key = music21.key.Key(state['key'].split()[0])
        scale = key.getScale()
        
        for i in range(16):
            note = music21.note.Note(scale.pitches[i % len(scale.pitches)])
            note.quarterLength = 1.0
            part.append(note)
        
        return part
    
    def _create_fallback_composition(self, state: MusicState) -> Dict:
        """Create a fallback composition when parsing fails"""
        score = music21.stream.Score()
        
        # Simple melody based on key
        key = state['key'].split()[0]
        if 'minor' in state['key']:
            scale = music21.scale.MinorScale(key)
        else:
            scale = music21.scale.MajorScale(key)
        
        melody = music21.stream.Part()
        for i in range(16):
            note = music21.note.Note(scale.pitches[i % 7])
            note.quarterLength = 1.0
            melody.append(note)
        
        score.append(melody)
        
        # Save files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_midi:
            score.write('midi', temp_midi.name)
            midi_path = temp_midi.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml") as temp_xml:
            score.write('musicxml', temp_xml.name)
            xml_path = temp_xml.name
        
        return {
            "midi_file": midi_path,
            "music_xml": xml_path,
            "composition": "Fallback composition generated"
        }

# Main Music Composition System
class AdvancedMusicCompositor:
    def __init__(self, groq_api_key: str):
        self.config = MusicConfig()
        self.llm = MusicLLM(groq_api_key)
        self.nodes = MusicNodes(self.llm, self.config)
        self.converter = MusicConverter()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the enhanced LangGraph workflow"""
        workflow = StateGraph(MusicState)
        
        # Add nodes
        workflow.add_node("input_processor", self.nodes.input_processor)
        workflow.add_node("structure_planner", self.nodes.structure_planner)
        workflow.add_node("melody_generator", self.nodes.melody_generator)
        workflow.add_node("harmony_creator", self.nodes.harmony_creator)
        workflow.add_node("rhythm_designer", self.nodes.rhythm_designer)
        workflow.add_node("style_integrator", self.nodes.style_integrator)
        workflow.add_node("music_converter", self.converter.create_composition)
        
        # Define workflow
        workflow.set_entry_point("input_processor")
        
        workflow.add_edge("input_processor", "structure_planner")
        workflow.add_edge("structure_planner", "melody_generator")
        workflow.add_edge("melody_generator", "harmony_creator")
        workflow.add_edge("harmony_creator", "rhythm_designer")
        workflow.add_edge("rhythm_designer", "style_integrator")
        workflow.add_edge("style_integrator", "music_converter")
        workflow.add_edge("music_converter", END)
        
        return workflow.compile()
    
    def compose(self, user_input: str, style: str = "classical") -> Dict:
        """Main composition method"""
        inputs = {
            "musician_input": user_input,
            "style": style
        }
        
        result = self.workflow.invoke(inputs)
        return result
    
    def visualize_workflow(self):
        """Display the workflow graph"""
        display(
            Image(
                self.workflow.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )

# Enhanced Playback and Analysis
class MusicAnalyzer:
    @staticmethod
    def analyze_composition(midi_file: str) -> Dict:
        """Analyze the generated composition"""
        try:
            score = music21.converter.parse(midi_file)
            
            analysis = {
                "key": str(score.analyze('key')),
                "time_signature": str(score.getTimeSignatures()[0]) if score.getTimeSignatures() else "4/4",
                "duration": score.duration.quarterLength,
                "note_count": len(score.flat.notes),
                "range": str(score.flat.notes[-1].pitch - score.flat.notes[0].pitch) if len(score.flat.notes) > 1 else "N/A"
            }
            
            return analysis
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def convert_to_audio(midi_file: str, output_wav: str = "output.wav"):
        """Convert MIDI to WAV using fluidsynth"""
        try:
            # Install fluidsynth if not available
            try:
                import fluidsynth
            except ImportError:
                print("Installing fluidsynth...")
                os.system("apt install fluidsynth -qq")
                os.system("cp /usr/share/sounds/sf2/FluidR3_GM.sf2 ./font.sf2")
            
            # Convert MIDI to WAV
            os.system(f"fluidsynth -ni font.sf2 {midi_file} -F {output_wav} -r 44100")
            return output_wav
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return None

# Example usage
def main():
    # Initialize the compositor
    compositor = AdvancedMusicCompositor("your_groq_api_key_here")
    
    # Visualize workflow
    compositor.visualize_workflow()
    
    # Create a composition
    user_request = "Write a joyful string quartet in G major with a moderate complexity"
    result = compositor.compose(user_request, style="classical")
    
    print(f"MIDI file: {result['midi_file']}")
    print(f"MusicXML file: {result['music_xml']}")
    
    # Analyze the composition
    analysis = MusicAnalyzer.analyze_composition(result['midi_file'])
    print("Composition Analysis:", analysis)
    
    # Convert to audio and play
    wav_file = MusicAnalyzer.convert_to_audio(result['midi_file'])
    if wav_file:
        display(Audio(wav_file))
    
    return result

# Run if in notebook environment
if __name__ == "__main__":
    result = main()
