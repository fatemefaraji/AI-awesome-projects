# Enhanced Music Composition System with Advanced Features

!pip install langchain langchain_core langgraph langchain_community langchain_groq music21 pretty_midi pygame
!apt install fluidsynth -qq

import os
from typing import TypedDict, Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image, Audio, Markdown
import music21
import tempfile
import random
import json
from datetime import datetime
import base64
import pretty_midi
import pygame
import io

# Enhanced State Definition with Comprehensive Musical Parameters
class MusicState(TypedDict):
    musician_input: str
    key: str
    tempo: int
    time_signature: str
    instruments: List[str]
    style: str
    emotion: str
    complexity: str
    duration_bars: int
    structure: Dict[str, Any]
    melody: str
    harmony: str
    rhythm: str
    bass_line: str
    counterpoint: str
    dynamics: str
    composition: str
    midi_file: str
    music_xml: str
    audio_file: str
    metadata: Dict[str, Any]
    analysis: Dict[str, Any]

# Configuration Management
class MusicConfig:
    def __init__(self):
        self.available_keys = [
            'C major', 'C minor', 'G major', 'G minor', 'D major', 'D minor',
            'A major', 'A minor', 'E major', 'E minor', 'F major', 'F minor',
            'Bb major', 'Bb minor', 'Eb major', 'Eb minor'
        ]
        self.available_tempos = {
            'largo': 50, 'adagio': 70, 'andante': 90, 'moderato': 110,
            'allegro': 130, 'presto': 160, 'prestissimo': 180
        }
        self.available_instruments = {
            'piano': music21.instrument.Piano,
            'violin': music21.instrument.Violin,
            'viola': music21.instrument.Viola,
            'cello': music21.instrument.Cello,
            'flute': music21.instrument.Flute,
            'clarinet': music21.instrument.Clarinet,
            'guitar': music21.instrument.Guitar,
            'string_quartet': [music21.instrument.Violin, music21.instrument.Violin, 
                             music21.instrument.Viola, music21.instrument.Cello]
        }
        self.available_styles = [
            'baroque', 'classical', 'romantic', 'impressionist', 
            'contemporary', 'jazz', 'minimalist', 'film_score'
        ]
        self.available_emotions = [
            'joyful', 'sorrowful', 'mysterious', 'energetic', 'calm', 
            'dramatic', 'heroic', 'nostalgic', 'tension', 'resolution'
        ]
        self.complexity_levels = ['simple', 'moderate', 'complex', 'virtuoso']

# Advanced Input Parser
class AdvancedInputParser:
    def __init__(self, config: MusicConfig):
        self.config = config
    
    def parse_musical_input(self, user_input: str) -> Dict[str, Any]:
        """Advanced parsing of musical instructions with context awareness"""
        input_lower = user_input.lower()
        parsed = {
            'key': self._extract_key(input_lower),
            'tempo': self._extract_tempo(input_lower),
            'style': self._extract_style(input_lower),
            'emotion': self._extract_emotion(input_lower),
            'instruments': self._extract_instruments(input_lower),
            'complexity': self._extract_complexity(input_lower),
            'duration_bars': self._extract_duration(input_lower),
            'time_signature': self._extract_time_signature(input_lower)
        }
        return parsed
    
    def _extract_key(self, text: str) -> str:
        # Enhanced key detection with modulations
        key_mappings = {
            'c major': 'C major', 'c minor': 'C minor',
            'g major': 'G major', 'g minor': 'G minor', 
            'd major': 'D major', 'd minor': 'D minor',
            'a major': 'A major', 'a minor': 'A minor',
            'e major': 'E major', 'e minor': 'E minor',
            'f major': 'F major', 'f minor': 'F minor'
        }
        
        for key_phrase, key_name in key_mappings.items():
            if key_phrase in text:
                return key_name
        return random.choice(self.config.available_keys)
    
    def _extract_tempo(self, text: str) -> int:
        tempo_words = {
            'largo': 50, 'adagio': 70, 'andante': 90, 'moderato': 110,
            'allegro': 130, 'presto': 160, 'prestissimo': 180,
            'slow': 70, 'medium': 110, 'fast': 150, 'very fast': 180
        }
        
        for word, tempo in tempo_words.items():
            if word in text:
                return tempo
        return 120  # Default moderate tempo
    
    def _extract_style(self, text: str) -> str:
        for style in self.config.available_styles:
            if style in text:
                return style
        return 'classical'
    
    def _extract_emotion(self, text: str) -> str:
        emotion_mappings = {
            'joyful': 'joyful', 'happy': 'joyful', 'cheerful': 'joyful',
            'sorrowful': 'sorrowful', 'sad': 'sorrowful', 'melancholy': 'sorrowful',
            'mysterious': 'mysterious', 'mystery': 'mysterious',
            'energetic': 'energetic', 'energetic': 'energetic', 'lively': 'energetic',
            'calm': 'calm', 'peaceful': 'calm', 'serene': 'calm',
            'dramatic': 'dramatic', 'drama': 'dramatic',
            'heroic': 'heroic', 'brave': 'heroic',
            'nostalgic': 'nostalgic', 'longing': 'nostalgic'
        }
        
        for word, emotion in emotion_mappings.items():
            if word in text:
                return emotion
        return 'calm'
    
    def _extract_instruments(self, text: str) -> List[str]:
        instruments_found = []
        for instrument in self.config.available_instruments.keys():
            if instrument in text:
                instruments_found.append(instrument)
        
        if not instruments_found:
            # Default based on style
            if 'quartet' in text:
                return ['string_quartet']
            elif 'solo' in text:
                return ['piano']
            else:
                return ['piano']
        return instruments_found
    
    def _extract_complexity(self, text: str) -> str:
        if any(word in text for word in ['simple', 'easy', 'basic']):
            return 'simple'
        elif any(word in text for word in ['complex', 'complicated', 'virtuoso', 'difficult']):
            return 'complex'
        elif any(word in text for word in ['moderate', 'medium', 'standard']):
            return 'moderate'
        return 'moderate'
    
    def _extract_duration(self, text: str) -> int:
        # Extract approximate duration in bars
        if 'short' in text:
            return 8
        elif 'long' in text or 'extended' in text:
            return 32
        elif 'medium' in text:
            return 16
        return 16  # Default
    
    def _extract_time_signature(self, text: str) -> str:
        if 'waltz' in text or '3/4' in text:
            return '3/4'
        elif '6/8' in text:
            return '6/8'
        return '4/4'

# Enhanced LLM Manager with Template Management
class AdvancedMusicLLM:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            temperature=0.8,  # Slightly higher for creativity
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        return {
            'structure': """
            Plan a musical structure for a {style} style composition with these parameters:
            - Key: {key}
            - Emotion: {emotion}
            - Complexity: {complexity}
            - Duration: {duration_bars} bars
            - Instruments: {instruments}
            - Time Signature: {time_signature}
            
            Provide a detailed JSON structure with:
            - sections (names and purposes)
            - section_lengths in measures
            - harmonic_progression by section
            - dynamic_plan (pp, p, mp, mf, f, ff)
            - texture_description
            - formal_structure (e.g., ABA, Sonata, Rondo)
            - development_plan
            
            Return ONLY valid JSON.
            """,
            
            'melody': """
            Compose a {emotion} melody in {key} for {style} style.
            Structure: {structure}
            Complexity: {complexity}
            Time Signature: {time_signature}
            Target Instrument: {primary_instrument}
            
            Create a melody that:
            1. Fits the emotional character
            2. Uses appropriate motifs and development
            3. Has clear phrasing and cadences
            4. Matches the complexity level
            
            Provide the melody in music21 format with:
            - Note sequences with rhythms
            - Articulations (staccato, legato, accents)
            - Dynamic markings
            - Phrasing slurs
            - Ornamentation if appropriate
            """,
            
            'harmony': """
            Create sophisticated harmonic progression for this {style} composition:
            Key: {key}
            Melody: {melody}
            Structure: {structure}
            Emotion: {emotion}
            
            Generate chords that:
            1. Support and enhance the melody
            2. Create appropriate tension and release
            3. Use style-appropriate harmonic language
            4. Include interesting voice leading
            5. Feature appropriate cadences
            
            Provide chords in music21 format with:
            - Chord symbols and inversions
            - Voice leading suggestions
            - Harmonic rhythm patterns
            - Modulation plans if applicable
            """,
            
            'rhythm': """
            Design rhythmic patterns for {style} composition:
            Time Signature: {time_signature}
            Tempo: {tempo}
            Melody: {melody}
            Harmony: {harmony}
            Emotion: {emotion}
            Instruments: {instruments}
            
            Create rhythm that:
            1. Supports the emotional character
            2. Uses style-appropriate patterns
            3. Creates interest through variation
            4. Coordinates between parts
            5. Includes appropriate syncopation
            
            Provide detailed rhythm specification.
            """,
            
            'bass_line': """
            Create a bass line for this composition:
            Key: {key}
            Harmony: {harmony}
            Style: {style}
            Instruments: {instruments}
            
            The bass line should:
            1. Establish harmonic foundation
            2. Create rhythmic interest
            3. Support the melody
            4. Use appropriate walking patterns or figures
            """,
            
            'counterpoint': """
            Add contrapuntal elements to this composition:
            Main Melody: {melody}
            Harmony: {harmony}
            Style: {style}
            
            Create counterpoint that:
            1. Follows style-appropriate rules
            2. Creates interesting interplay
            3. Supports harmonic progression
            4. Maintains independence of lines
            """,
            
            'integration': """
            Integrate all musical elements into a cohesive {style} composition:
            
            Structure: {structure}
            Melody: {melody}
            Harmony: {harmony}
            Rhythm: {rhythm}
            Bass Line: {bass_line}
            Counterpoint: {counterpoint}
            
            Additional Parameters:
            - Key: {key}
            - Tempo: {tempo}
            - Time Signature: {time_signature}
            - Emotion: {emotion}
            - Instruments: {instruments}
            
            Create a complete music21 score with:
            - Proper instrumentation
            - Detailed dynamics and articulations
            - Phrasing and expression marks
            - Tempo variations if appropriate
            - Performance instructions
            """
        }
    
    def generate(self, template_name: str, **kwargs) -> str:
        prompt = ChatPromptTemplate.from_template(self.templates[template_name])
        chain = prompt | self.llm
        response = chain.invoke(kwargs)
        return response.content

# Advanced Music Nodes
class AdvancedMusicNodes:
    def __init__(self, llm: AdvancedMusicLLM, config: MusicConfig):
        self.llm = llm
        self.config = config
        self.parser = AdvancedInputParser(config)
    
    def input_analysis(self, state: MusicState) -> Dict:
        """Enhanced input processing with musical intelligence"""
        parsed_input = self.parser.parse_musical_input(state["musician_input"])
        
        return {
            "key": parsed_input["key"],
            "tempo": parsed_input["tempo"],
            "style": parsed_input["style"],
            "emotion": parsed_input["emotion"],
            "instruments": parsed_input["instruments"],
            "complexity": parsed_input["complexity"],
            "duration_bars": parsed_input["duration_bars"],
            "time_signature": parsed_input["time_signature"],
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "input_parameters": parsed_input,
                "version": "2.0",
                "workflow_id": f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        }
    
    def structural_design(self, state: MusicState) -> Dict:
        """Design comprehensive musical structure"""
        try:
            structure_json = self.llm.generate('structure', **state)
            structure_data = json.loads(structure_json)
        except Exception as e:
            print(f"Structure generation failed, using fallback: {e}")
            structure_data = self._create_fallback_structure(state)
        
        return {"structure": structure_data}
    
    def _create_fallback_structure(self, state: MusicState) -> Dict:
        """Create fallback musical structure"""
        base_structure = {
            "sections": ["Exposition", "Development", "Recapitulation"],
            "section_lengths": [8, 8, 8],
            "harmonic_progression": {
                "Exposition": ["I", "IV", "V", "I"],
                "Development": ["vi", "ii", "V", "I"],
                "Recapitulation": ["I", "IV", "V", "I"]
            },
            "dynamic_plan": ["p", "crescendo", "f", "diminuendo"],
            "texture_description": "Melody with harmonic accompaniment",
            "formal_structure": "ABA",
            "development_plan": "Thematic development with modulation"
        }
        return base_structure
    
    def melodic_composition(self, state: MusicState) -> Dict:
        """Compose sophisticated melody"""
        primary_instrument = state["instruments"][0] if state["instruments"] else "piano"
        
        melody = self.llm.generate('melody', 
            primary_instrument=primary_instrument,
            **state
        )
        return {"melody": melody}
    
    def harmonic_development(self, state: MusicState) -> Dict:
        """Develop advanced harmonic progression"""
        harmony = self.llm.generate('harmony', **state)
        return {"harmony": harmony}
    
    def rhythmic_design(self, state: MusicState) -> Dict:
        """Design complex rhythmic patterns"""
        rhythm = self.llm.generate('rhythm', **state)
        return {"rhythm": rhythm}
    
    def bass_construction(self, state: MusicState) -> Dict:
        """Construct bass line"""
        bass_line = self.llm.generate('bass_line', **state)
        return {"bass_line": bass_line}
    
    def contrapuntal_development(self, state: MusicState) -> Dict:
        """Add counterpoint"""
        counterpoint = self.llm.generate('counterpoint', **state)
        return {"counterpoint": counterpoint}
    
    def dynamic_planning(self, state: MusicState) -> Dict:
        """Plan dynamics and expression"""
        # For now, create basic dynamics plan
        # In a more advanced version, this could use LLM
        dynamics_plan = {
            "overall_shape": "crescendo to climax then diminuendo",
            "section_dynamics": ["p", "mp", "mf", "f", "mp", "p"],
            "articulations": ["legato", "staccato", "accent"],
            "expression_marks": ["espressivo", "dolce", "agitato"]
        }
        return {"dynamics": json.dumps(dynamics_plan)}
    
    def compositional_integration(self, state: MusicState) -> Dict:
        """Integrate all musical elements"""
        composition = self.llm.generate('integration', **state)
        return {"composition": composition}

# Advanced Music Converter with Multiple Output Formats
class AdvancedMusicConverter:
    def __init__(self):
        self.instrument_mapping = {
            'piano': music21.instrument.Piano,
            'violin': music21.instrument.Violin,
            'viola': music21.instrument.Viola,
            'cello': music21.instrument.Cello,
            'flute': music21.instrument.Flute,
            'clarinet': music21.instrument.Clarinet,
            'guitar': music21.instrument.Guitar
        }
    
    def create_comprehensive_score(self, state: MusicState) -> Dict:
        """Create advanced musical score with multiple formats"""
        try:
            score = music21.stream.Score()
            
            # Add metadata
            metadata = music21.metadata.Metadata()
            metadata.title = f"{state['style'].title()} Composition in {state['key']}"
            metadata.composer = "AI Music Composer v2.0"
            metadata.date = datetime.now().strftime("%Y")
            score.insert(0, metadata)
            
            # Add tempo
            tempo = music21.tempo.MetronomeMark(number=state['tempo'])
            score.insert(0, tempo)
            
            # Add time signature
            time_sig = music21.meter.TimeSignature(state['time_signature'])
            score.insert(0, time_sig)
            
            # Add key signature
            key_sig = music21.key.Key(state['key'].split()[0])
            if 'minor' in state['key']:
                key_sig = key_sig.minor
            score.insert(0, key_sig)
            
            # Create parts for each instrument
            for instrument_name in state['instruments']:
                part = self._create_advanced_instrument_part(instrument_name, state)
                score.insert(0, part)
            
            # Save MIDI
            midi_path = self._save_midi(score)
            # Save MusicXML
            xml_path = self._save_musicxml(score)
            # Generate audio
            audio_path = self._generate_audio(midi_path)
            # Analyze composition
            analysis = self._analyze_composition(score)
            
            return {
                "midi_file": midi_path,
                "music_xml": xml_path,
                "audio_file": audio_path,
                "analysis": analysis,
                "composition": "Advanced composition successfully generated"
            }
            
        except Exception as e:
            print(f"Advanced composition failed: {e}")
            return self._create_advanced_fallback(state)
    
    def _create_advanced_instrument_part(self, instrument_name: str, state: MusicState) -> music21.stream.Part:
        """Create sophisticated instrument part"""
        part = music21.stream.Part()
        
        # Add instrument
        if instrument_name in self.instrument_mapping:
            instrument = self.instrument_mapping[instrument_name]()
            part.insert(0, instrument)
        
        # Create more sophisticated musical content based on parameters
        key = music21.key.Key(state['key'].split()[0])
        if 'minor' in state['key']:
            scale = key.getScale('harmonic minor')
        else:
            scale = key.getScale('major')
        
        # Generate musical content based on complexity
        if state['complexity'] == 'simple':
            notes = self._generate_simple_melody(scale, state['duration_bars'])
        elif state['complexity'] == 'complex':
            notes = self._generate_complex_melody(scale, state['duration_bars'])
        else:
            notes = self._generate_moderate_melody(scale, state['duration_bars'])
        
        for note in notes:
            part.append(note)
        
        return part
    
    def _generate_simple_melody(self, scale, duration_bars):
        """Generate simple stepwise melody"""
        notes = []
        for i in range(duration_bars * 4):  # 4 notes per bar
            note = music21.note.Note(scale.pitches[i % len(scale.pitches)])
            note.quarterLength = 1.0
            notes.append(note)
        return notes
    
    def _generate_moderate_melody(self, scale, duration_bars):
        """Generate moderate complexity melody"""
        notes = []
        rhythms = [0.5, 1.0, 2.0]
        for i in range(duration_bars * 4):
            note = music21.note.Note(scale.pitches[i % len(scale.pitches)])
            note.quarterLength = random.choice(rhythms)
            # Add some articulation
            if random.random() > 0.7:
                note.articulations.append(music21.articulation.Staccato())
            notes.append(note)
        return notes
    
    def _generate_complex_melody(self, scale, duration_bars):
        """Generate complex melody with variations"""
        notes = []
        rhythms = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        for i in range(duration_bars * 8):  # More notes for complexity
            pitch_index = (i + random.randint(-1, 1)) % len(scale.pitches)
            note = music21.note.Note(scale.pitches[pitch_index])
            note.quarterLength = random.choice(rhythms)
            
            # Add articulations and expressions
            if random.random() > 0.6:
                note.articulations.append(random.choice([
                    music21.articulation.Staccato(),
                    music21.articulation.Accent(),
                    music21.articulation.Tenuto()
                ]))
            
            notes.append(note)
        return notes
    
    def _save_midi(self, score) -> str:
        """Save score as MIDI"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_midi:
            score.write('midi', temp_midi.name)
            return temp_midi.name
    
    def _save_musicxml(self, score) -> str:
        """Save score as MusicXML"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml") as temp_xml:
            score.write('musicxml', temp_xml.name)
            return temp_xml.name
    
    def _generate_audio(self, midi_path: str) -> str:
        """Generate audio from MIDI"""
        try:
            output_wav = "enhanced_output.wav"
            # Ensure fluidsynth is available
            os.system("apt install fluidsynth -qq > /dev/null 2>&1")
            os.system("cp /usr/share/sounds/sf2/FluidR3_GM.sf2 ./font.sf2 2>/dev/null || true")
            
            # Convert MIDI to WAV
            os.system(f"fluidsynth -ni font.sf2 {midi_path} -F {output_wav} -r 44100 > /dev/null 2>&1")
            return output_wav
        except Exception as e:
            print(f"Audio generation failed: {e}")
            return ""
    
    def _analyze_composition(self, score) -> Dict:
        """Analyze the generated composition"""
        try:
            analysis = {
                "key": str(score.analyze('key')),
                "time_signature": str(score.getTimeSignatures()[0]) if score.getTimeSignatures() else "4/4",
                "duration_quarters": score.duration.quarterLength,
                "note_count": len(score.flat.notes),
                "range": self._calculate_range(score),
                "average_note_length": self._calculate_average_note_length(score),
                "instrumentation": [str(instr) for instr in score.parts]
            }
            return analysis
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_range(self, score):
        """Calculate pitch range of composition"""
        notes = score.flat.notes
        if len(notes) > 1:
            pitches = [n.pitch.ps for n in notes if hasattr(n, 'pitch')]
            return max(pitches) - min(pitches) if pitches else 0
        return 0
    
    def _calculate_average_note_length(self, score):
        """Calculate average note duration"""
        notes = score.flat.notes
        if notes:
            total_length = sum(n.duration.quarterLength for n in notes)
            return total_length / len(notes)
        return 0
    
    def _create_advanced_fallback(self, state: MusicState) -> Dict:
        """Create advanced fallback composition"""
        score = music21.stream.Score()
        
        # Create a more sophisticated fallback
        for instrument_name in state['instruments']:
            part = self._create_advanced_instrument_part(instrument_name, state)
            score.append(part)
        
        midi_path = self._save_midi(score)
        xml_path = self._save_musicxml(score)
        audio_path = self._generate_audio(midi_path)
        
        return {
            "midi_file": midi_path,
            "music_xml": xml_path,
            "audio_file": audio_path,
            "analysis": {"fallback": "Advanced generation failed, using sophisticated fallback"},
            "composition": "Sophisticated fallback composition generated"
        }

# Main Advanced Music Composition System
class AdvancedMusicCompositor:
    def __init__(self, groq_api_key: str):
        self.config = MusicConfig()
        self.llm = AdvancedMusicLLM(groq_api_key)
        self.nodes = AdvancedMusicNodes(self.llm, self.config)
        self.converter = AdvancedMusicConverter()
        self.workflow = self._build_advanced_workflow()
    
    def _build_advanced_workflow(self) -> StateGraph:
        """Build comprehensive musical workflow"""
        workflow = StateGraph(MusicState)
        
        # Add all enhanced nodes
        workflow.add_node("input_analysis", self.nodes.input_analysis)
        workflow.add_node("structural_design", self.nodes.structural_design)
        workflow.add_node("melodic_composition", self.nodes.melodic_composition)
        workflow.add_node("harmonic_development", self.nodes.harmonic_development)
        workflow.add_node("rhythmic_design", self.nodes.rhythmic_design)
        workflow.add_node("bass_construction", self.nodes.bass_construction)
        workflow.add_node("contrapuntal_development", self.nodes.contrapuntal_development)
        workflow.add_node("dynamic_planning", self.nodes.dynamic_planning)
        workflow.add_node("compositional_integration", self.nodes.compositional_integration)
        workflow.add_node("music_conversion", self.converter.create_comprehensive_score)
        
        # Define sophisticated workflow
        workflow.set_entry_point("input_analysis")
        
        workflow.add_edge("input_analysis", "structural_design")
        workflow.add_edge("structural_design", "melodic_composition")
        workflow.add_edge("melodic_composition", "harmonic_development")
        workflow.add_edge("harmonic_development", "rhythmic_design")
        workflow.add_edge("rhythmic_design", "bass_construction")
        workflow.add_edge("bass_construction", "contrapuntal_development")
        workflow.add_edge("contrapuntal_development", "dynamic_planning")
        workflow.add_edge("dynamic_planning", "compositional_integration")
        workflow.add_edge("compositional_integration", "music_conversion")
        workflow.add_edge("music_conversion", END)
        
        return workflow.compile()
    
    def compose(self, user_input: str, style: str = None) -> Dict:
        """Main composition method with enhanced capabilities"""
        inputs = {"musician_input": user_input}
        if style:
            inputs["style"] = style
        
        print("🎵 Composing music...")
        result = self.workflow.invoke(inputs)
        print("✅ Composition complete!")
        
        return result
    
    def visualize_workflow(self):
        """Display the advanced workflow"""
        display(Markdown("### Advanced Music Composition Workflow"))
        display(
            Image(
                self.workflow.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )
    
    def play_composition(self, result: Dict):
        """Play the generated composition"""
        if result.get('audio_file') and os.path.exists(result['audio_file']):
            display(Audio(result['audio_file']))
        else:
            print("Audio file not available for playback")
    
    def display_analysis(self, result: Dict):
        """Display composition analysis"""
        analysis = result.get('analysis', {})
        display(Markdown("### Composition Analysis"))
        for key, value in analysis.items():
            display(Markdown(f"**{key.replace('_', ' ').title()}:** {value}"))

# Example usage and demonstration
def demonstrate_enhanced_composer():
    """Demonstrate the enhanced music composition system"""
    
    # Initialize the advanced composer
    composer = AdvancedMusicCompositor("0000000000000000000000")
    
    # Display workflow
    composer.visualize_workflow()
    
    # Example compositions
    examples = [
        "Compose a joyful piano sonata in G major with virtuoso complexity",
        "Create a mysterious string quartet in D minor with moderate complexity",
        "Write a dramatic film score in C minor for full orchestra",
        "Compose a calm flute solo in F major with simple melody"
    ]
    
    for i, example in enumerate(examples[:1]):  # Just run first example for demonstration
        print(f"\n🎼 Example {i+1}: {example}")
        print("=" * 60)
        
        result = composer.compose(example)
        
        # Display results
        print(f"📁 MIDI File: {result['midi_file']}")
        print(f"📄 MusicXML: {result['music_xml']}")
        if result.get('audio_file'):
            print(f"🔊 Audio File: {result['audio_file']}")
        
        # Show analysis
        composer.display_analysis(result)
        
        # Play composition
        composer.play_composition(result)
        
        print("\n" + "=" * 60)
    
    return composer

# Run the enhanced demonstration
if __name__ == "__main__":
    advanced_composer = demonstrate_enhanced_composer()
