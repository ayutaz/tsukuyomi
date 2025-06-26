#!/usr/bin/env python3
"""MOS (Mean Opinion Score) Evaluation Tool for TTS

This tool provides:
- Web interface for subjective evaluation
- Automated test set generation
- Statistical analysis of results
- Inter-rater reliability metrics
- Export functionality for results
"""

import argparse
import json
import logging
import os
import random
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import streamlit as st
from scipy import stats
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MOSDatabase:
    """Database handler for MOS evaluation results"""

    def __init__(self, db_path: str = "mos_evaluation.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                evaluator_id TEXT NOT NULL,
                audio_id TEXT NOT NULL,
                score INTEGER NOT NULL,
                naturalness INTEGER,
                intelligibility INTEGER,
                speaker_similarity INTEGER,
                emotion_appropriateness INTEGER,
                overall_quality INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_ms INTEGER,
                comments TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_samples (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                text TEXT NOT NULL,
                system_name TEXT NOT NULL,
                speaker_id TEXT,
                emotion TEXT,
                reference_path TEXT,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluators (
                id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                experience_level TEXT,
                native_language TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def add_evaluator(
        self,
        name: str,
        email: str = None,
        experience: str = "beginner",
        language: str = "ja",
    ) -> str:
        """Add new evaluator"""
        evaluator_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO evaluators (id, name, email, experience_level, native_language)
            VALUES (?, ?, ?, ?, ?)
        """,
            (evaluator_id, name, email, experience, language),
        )

        conn.commit()
        conn.close()

        return evaluator_id

    def add_audio_sample(
        self,
        file_path: str,
        text: str,
        system_name: str,
        speaker_id: str = None,
        emotion: str = None,
        reference_path: str = None,
        metadata: Dict = None,
    ) -> str:
        """Add audio sample for evaluation"""
        audio_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO audio_samples 
            (id, file_path, text, system_name, speaker_id, emotion, reference_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                audio_id,
                file_path,
                text,
                system_name,
                speaker_id,
                emotion,
                reference_path,
                json.dumps(metadata) if metadata else None,
            ),
        )

        conn.commit()
        conn.close()

        return audio_id

    def add_evaluation(
        self,
        evaluator_id: str,
        audio_id: str,
        scores: Dict,
        duration_ms: int = None,
        comments: str = None,
    ):
        """Add evaluation result"""
        eval_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO evaluations 
            (id, evaluator_id, audio_id, score, naturalness, intelligibility,
             speaker_similarity, emotion_appropriateness, overall_quality,
             duration_ms, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                eval_id,
                evaluator_id,
                audio_id,
                scores.get("mos", 3),
                scores.get("naturalness", 3),
                scores.get("intelligibility", 3),
                scores.get("speaker_similarity", 3),
                scores.get("emotion_appropriateness", 3),
                scores.get("overall_quality", 3),
                duration_ms,
                comments,
            ),
        )

        conn.commit()
        conn.close()

    def get_results(self, system_name: Optional[str] = None) -> pd.DataFrame:
        """Get evaluation results"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT 
                e.*,
                a.system_name,
                a.text,
                a.speaker_id,
                a.emotion,
                ev.name as evaluator_name,
                ev.experience_level
            FROM evaluations e
            JOIN audio_samples a ON e.audio_id = a.id
            JOIN evaluators ev ON e.evaluator_id = ev.id
        """

        if system_name:
            query += f" WHERE a.system_name = '{system_name}'"

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df


class MOSEvaluator:
    """MOS evaluation manager"""

    def __init__(self, db_path: str = "mos_evaluation.db"):
        self.db = MOSDatabase(db_path)

    def create_evaluation_set(
        self,
        audio_files: List[Dict[str, str]],
        output_dir: Path,
        samples_per_system: int = 10,
        include_references: bool = True,
    ) -> Dict:
        """Create evaluation set with randomization"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group by system
        systems = {}
        for audio in audio_files:
            system = audio["system"]
            if system not in systems:
                systems[system] = []
            systems[system].append(audio)

        # Sample from each system
        evaluation_set = []
        for system, files in systems.items():
            sampled = random.sample(files, min(samples_per_system, len(files)))
            evaluation_set.extend(sampled)

        # Randomize order
        random.shuffle(evaluation_set)

        # Add to database
        for i, audio in enumerate(evaluation_set):
            audio_id = self.db.add_audio_sample(
                file_path=audio["path"],
                text=audio["text"],
                system_name=audio["system"],
                speaker_id=audio.get("speaker_id"),
                emotion=audio.get("emotion"),
                reference_path=audio.get("reference"),
            )
            evaluation_set[i]["audio_id"] = audio_id

        # Save evaluation set configuration
        config = {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(evaluation_set),
            "systems": list(systems.keys()),
            "samples_per_system": samples_per_system,
            "evaluation_set": evaluation_set,
        }

        with open(output_dir / "evaluation_config.json", "w") as f:
            json.dump(config, f, indent=2)

        return config

    def calculate_statistics(self, system_name: Optional[str] = None) -> Dict:
        """Calculate MOS statistics"""
        df = self.db.get_results(system_name)

        if df.empty:
            return {}

        # Overall statistics
        stats = {
            "overall": {
                "mean": df["score"].mean(),
                "std": df["score"].std(),
                "median": df["score"].median(),
                "n_evaluations": len(df),
                "n_evaluators": df["evaluator_id"].nunique(),
                "n_samples": df["audio_id"].nunique(),
            }
        }

        # Per-system statistics
        system_stats = {}
        for system in df["system_name"].unique():
            system_df = df[df["system_name"] == system]
            system_stats[system] = {
                "mean": system_df["score"].mean(),
                "std": system_df["score"].std(),
                "median": system_df["score"].median(),
                "ci_95": self._confidence_interval(system_df["score"]),
                "n_evaluations": len(system_df),
                "aspects": {
                    "naturalness": system_df["naturalness"].mean(),
                    "intelligibility": system_df["intelligibility"].mean(),
                    "speaker_similarity": system_df["speaker_similarity"].mean(),
                    "emotion_appropriateness": system_df[
                        "emotion_appropriateness"
                    ].mean(),
                    "overall_quality": system_df["overall_quality"].mean(),
                },
            }
        stats["per_system"] = system_stats

        # Inter-rater reliability
        if df["evaluator_id"].nunique() > 1:
            stats["inter_rater"] = self._calculate_inter_rater_reliability(df)

        # Statistical tests between systems
        if len(system_stats) > 1:
            stats["comparisons"] = self._compare_systems(df)

        return stats

    def _confidence_interval(self, scores, confidence=0.95):
        """Calculate confidence interval"""
        n = len(scores)
        if n < 2:
            return (scores.mean(), scores.mean())

        sem = stats.sem(scores)
        interval = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
        mean = scores.mean()

        return (mean - interval, mean + interval)

    def _calculate_inter_rater_reliability(self, df: pd.DataFrame) -> Dict:
        """Calculate inter-rater reliability metrics"""
        # Create evaluator-audio matrix
        pivot = df.pivot_table(
            values="score", index="audio_id", columns="evaluator_id", aggfunc="first"
        )

        # Remove samples with missing evaluations
        pivot = pivot.dropna()

        if pivot.empty or pivot.shape[1] < 2:
            return {}

        # Calculate metrics
        reliability = {}

        # Intraclass Correlation Coefficient (ICC)
        # Using one-way random effects model
        reliability["icc"] = self._calculate_icc(pivot.values)

        # Krippendorff's alpha
        reliability["krippendorff_alpha"] = self._calculate_krippendorff_alpha(pivot)

        # Pairwise agreement
        evaluators = pivot.columns.tolist()
        if len(evaluators) == 2:
            # Cohen's kappa for two raters
            reliability["cohen_kappa"] = cohen_kappa_score(
                pivot[evaluators[0]], pivot[evaluators[1]], weights="linear"
            )

        return reliability

    def _calculate_icc(self, ratings):
        """Calculate Intraclass Correlation Coefficient"""
        n_samples, n_raters = ratings.shape

        # Calculate mean squares
        row_means = np.mean(ratings, axis=1)
        grand_mean = np.mean(ratings)

        # Between-subject mean square
        ssb = n_raters * np.sum((row_means - grand_mean) ** 2)
        msb = ssb / (n_samples - 1)

        # Within-subject mean square
        ssw = np.sum((ratings - row_means[:, np.newaxis]) ** 2)
        msw = ssw / (n_samples * (n_raters - 1))

        # ICC(1,1) - one-way random effects
        icc = (msb - msw) / (msb + (n_raters - 1) * msw)

        return max(0, icc)  # ICC can be negative, but we floor at 0

    def _calculate_krippendorff_alpha(self, pivot):
        """Calculate Krippendorff's alpha"""
        # Simplified implementation
        # For full implementation, use krippendorff package
        values = pivot.values.flatten()
        values = values[~np.isnan(values)]

        if len(values) < 2:
            return 0

        # Calculate observed disagreement
        n = len(values)
        observed_disagreement = 0

        for i in range(n):
            for j in range(i + 1, n):
                observed_disagreement += (values[i] - values[j]) ** 2

        observed_disagreement /= n * (n - 1) / 2

        # Calculate expected disagreement
        expected_disagreement = np.var(values)

        # Alpha
        alpha = 1 - (observed_disagreement / expected_disagreement)

        return alpha

    def _compare_systems(self, df: pd.DataFrame) -> Dict:
        """Statistical comparison between systems"""
        systems = df["system_name"].unique()
        comparisons = {}

        for i, sys1 in enumerate(systems):
            for sys2 in systems[i + 1 :]:
                scores1 = df[df["system_name"] == sys1]["score"]
                scores2 = df[df["system_name"] == sys2]["score"]

                # T-test
                t_stat, p_value = stats.ttest_ind(scores1, scores2)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((scores1.var() + scores2.var()) / 2)
                cohen_d = (scores1.mean() - scores2.mean()) / pooled_std

                comparisons[f"{sys1}_vs_{sys2}"] = {
                    "mean_diff": scores1.mean() - scores2.mean(),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "cohen_d": cohen_d,
                    "significant": p_value < 0.05,
                }

        return comparisons

    def generate_report(self, output_path: str, format: str = "html"):
        """Generate evaluation report"""
        stats = self.calculate_statistics()
        df = self.db.get_results()

        if format == "html":
            self._generate_html_report(stats, df, output_path)
        elif format == "pdf":
            self._generate_pdf_report(stats, df, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_html_report(self, stats: Dict, df: pd.DataFrame, output_path: str):
        """Generate HTML report"""
        html = f"""
        <html>
        <head>
            <title>MOS Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>MOS Evaluation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Overall Statistics</h2>
            <div class="metric">
                <p>Total Evaluations: {stats['overall']['n_evaluations']}</p>
                <p>Number of Evaluators: {stats['overall']['n_evaluators']}</p>
                <p>Number of Samples: {stats['overall']['n_samples']}</p>
                <p>Overall MOS: {stats['overall']['mean']:.2f} ± {stats['overall']['std']:.2f}</p>
            </div>
            
            <h2>System Comparison</h2>
            <table>
                <tr>
                    <th>System</th>
                    <th>MOS</th>
                    <th>95% CI</th>
                    <th>Naturalness</th>
                    <th>Intelligibility</th>
                    <th>Overall Quality</th>
                </tr>
        """

        for system, data in stats["per_system"].items():
            html += f"""
                <tr>
                    <td>{system}</td>
                    <td>{data['mean']:.2f} ± {data['std']:.2f}</td>
                    <td>[{data['ci_95'][0]:.2f}, {data['ci_95'][1]:.2f}]</td>
                    <td>{data['aspects']['naturalness']:.2f}</td>
                    <td>{data['aspects']['intelligibility']:.2f}</td>
                    <td>{data['aspects']['overall_quality']:.2f}</td>
                </tr>
            """

        html += """
            </table>
            
            <h2>Inter-rater Reliability</h2>
        """

        if stats.get("inter_rater"):
            html += '<div class="metric">'
            for metric, value in stats["inter_rater"].items():
                html += f"<p>{metric}: {value:.3f}</p>"
            html += "</div>"
        else:
            html += "<p>Not enough data for inter-rater analysis</p>"

        html += """
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html)

    def plot_results(self, output_dir: Path):
        """Generate visualization plots"""
        df = self.db.get_results()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

        # 1. Box plot by system
        plt.figure()
        sns.boxplot(data=df, x="system_name", y="score")
        plt.title("MOS Distribution by System")
        plt.xlabel("System")
        plt.ylabel("MOS Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "mos_boxplot.png", dpi=300)
        plt.close()

        # 2. Aspect radar chart
        aspects = [
            "naturalness",
            "intelligibility",
            "speaker_similarity",
            "emotion_appropriateness",
            "overall_quality",
        ]

        systems = df["system_name"].unique()
        angles = np.linspace(0, 2 * np.pi, len(aspects), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        for system in systems:
            system_df = df[df["system_name"] == system]
            values = [system_df[aspect].mean() for aspect in aspects]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, label=system)
            ax.fill(angles, values, alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(aspects)
        ax.set_ylim(1, 5)
        ax.set_ylabel("Score")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        plt.title("System Performance by Aspect")
        plt.tight_layout()
        plt.savefig(output_dir / "aspect_radar.png", dpi=300)
        plt.close()

        # 3. Evaluator consistency
        if df["evaluator_id"].nunique() > 1:
            plt.figure()
            evaluator_means = df.groupby("evaluator_id")["score"].mean().sort_values()
            evaluator_names = [f"E{i+1}" for i in range(len(evaluator_means))]

            plt.barh(evaluator_names, evaluator_means)
            plt.xlabel("Average MOS Score")
            plt.ylabel("Evaluator")
            plt.title("Evaluator Rating Tendencies")
            plt.axvline(
                x=df["score"].mean(),
                color="r",
                linestyle="--",
                label=f'Overall Mean: {df["score"].mean():.2f}',
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "evaluator_consistency.png", dpi=300)
            plt.close()


def create_streamlit_app():
    """Create Streamlit web interface for MOS evaluation"""
    st.set_page_config(page_title="MOS Evaluation Tool", page_icon="🎵", layout="wide")

    st.title("🎵 MOS Evaluation Tool")
    st.markdown("### Subjective evaluation of text-to-speech systems")

    # Initialize session state
    if "evaluator_id" not in st.session_state:
        st.session_state.evaluator_id = None
    if "current_sample" not in st.session_state:
        st.session_state.current_sample = 0
    if "evaluation_data" not in st.session_state:
        st.session_state.evaluation_data = []

    # Sidebar for evaluator info
    with st.sidebar:
        st.header("👤 Evaluator Information")

        if st.session_state.evaluator_id is None:
            name = st.text_input("Name", key="evaluator_name")
            email = st.text_input("Email (optional)", key="evaluator_email")
            experience = st.selectbox(
                "Experience with TTS evaluation",
                ["beginner", "intermediate", "expert"],
                key="evaluator_experience",
            )
            language = st.selectbox(
                "Native Language", ["ja", "en", "zh", "other"], key="evaluator_language"
            )

            if st.button("Start Evaluation"):
                if name:
                    db = MOSDatabase()
                    st.session_state.evaluator_id = db.add_evaluator(
                        name, email, experience, language
                    )
                    st.success(f"Welcome, {name}!")
                else:
                    st.error("Please enter your name")
        else:
            st.success(f"Evaluator ID: {st.session_state.evaluator_id[:8]}...")
            if st.button("Switch Evaluator"):
                st.session_state.evaluator_id = None
                st.experimental_rerun()

    # Main evaluation interface
    if st.session_state.evaluator_id:
        # Load evaluation samples
        db = MOSDatabase()
        conn = sqlite3.connect(db.db_path)
        samples = pd.read_sql_query("SELECT * FROM audio_samples", conn)
        conn.close()

        if samples.empty:
            st.warning("No audio samples found. Please prepare evaluation set first.")
        else:
            # Current sample
            if st.session_state.current_sample < len(samples):
                sample = samples.iloc[st.session_state.current_sample]

                st.header(
                    f"Sample {st.session_state.current_sample + 1} of {len(samples)}"
                )

                # Display text
                st.markdown(f"**Text:** {sample['text']}")

                # Audio player
                col1, col2 = st.columns([3, 1])
                with col1:
                    audio_data, sr = sf.read(sample["file_path"])
                    st.audio(audio_data, sample_rate=sr)

                # Reference audio if available
                if pd.notna(sample["reference_path"]):
                    st.markdown("**Reference Audio:**")
                    ref_data, ref_sr = sf.read(sample["reference_path"])
                    st.audio(ref_data, sample_rate=ref_sr)

                # Evaluation form
                st.markdown("### Evaluation")

                col1, col2 = st.columns(2)

                with col1:
                    mos_score = st.slider(
                        "Overall MOS Score",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="1: Bad, 2: Poor, 3: Fair, 4: Good, 5: Excellent",
                    )

                    naturalness = st.slider(
                        "Naturalness",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="How natural does the speech sound?",
                    )

                    intelligibility = st.slider(
                        "Intelligibility",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="How clearly can you understand the words?",
                    )

                with col2:
                    speaker_similarity = st.slider(
                        "Speaker Similarity",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="How similar is it to the target speaker?",
                    )

                    emotion_appropriateness = st.slider(
                        "Emotion Appropriateness",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="How appropriate is the emotional expression?",
                    )

                    overall_quality = st.slider(
                        "Overall Quality",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Overall impression of the synthesis quality",
                    )

                comments = st.text_area("Comments (optional)")

                # Navigation
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    if st.button(
                        "Previous", disabled=st.session_state.current_sample == 0
                    ):
                        st.session_state.current_sample -= 1
                        st.experimental_rerun()

                with col2:
                    if st.button("Submit & Next"):
                        # Save evaluation
                        scores = {
                            "mos": mos_score,
                            "naturalness": naturalness,
                            "intelligibility": intelligibility,
                            "speaker_similarity": speaker_similarity,
                            "emotion_appropriateness": emotion_appropriateness,
                            "overall_quality": overall_quality,
                        }

                        db.add_evaluation(
                            st.session_state.evaluator_id,
                            sample["id"],
                            scores,
                            comments=comments if comments else None,
                        )

                        # Move to next sample
                        st.session_state.current_sample += 1
                        st.experimental_rerun()

                with col3:
                    progress = (st.session_state.current_sample + 1) / len(samples)
                    st.progress(progress)
                    st.text(f"{int(progress * 100)}% complete")

            else:
                st.success("🎉 Evaluation complete! Thank you for your participation.")

                if st.button("View Results"):
                    evaluator = MOSEvaluator()
                    stats = evaluator.calculate_statistics()

                    st.header("Evaluation Results")
                    st.json(stats)


def main():
    parser = argparse.ArgumentParser(description="MOS Evaluation Tool")
    parser.add_argument(
        "--mode",
        choices=["prepare", "evaluate", "analyze", "web"],
        required=True,
        help="Operation mode",
    )
    parser.add_argument(
        "--audio-dir", type=str, help="Directory containing audio files"
    )
    parser.add_argument("--config", type=str, help="Evaluation configuration file")
    parser.add_argument(
        "--output", type=str, default="mos_results", help="Output directory"
    )
    parser.add_argument(
        "--db", type=str, default="mos_evaluation.db", help="Database file path"
    )

    args = parser.parse_args()

    if args.mode == "prepare":
        # Prepare evaluation set
        if not args.audio_dir:
            parser.error("--audio-dir required for prepare mode")

        evaluator = MOSEvaluator(args.db)

        # Scan audio files
        audio_files = []
        audio_dir = Path(args.audio_dir)

        for system_dir in audio_dir.iterdir():
            if system_dir.is_dir():
                for audio_file in system_dir.glob("*.wav"):
                    # Extract metadata from filename or accompanying file
                    metadata_file = audio_file.with_suffix(".json")
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}

                    audio_files.append(
                        {
                            "path": str(audio_file),
                            "system": system_dir.name,
                            "text": metadata.get("text", "Unknown"),
                            "speaker_id": metadata.get("speaker_id"),
                            "emotion": metadata.get("emotion"),
                            "reference": metadata.get("reference"),
                        }
                    )

        # Create evaluation set
        config = evaluator.create_evaluation_set(
            audio_files, Path(args.output), samples_per_system=10
        )

        logger.info(f"Created evaluation set with {config['total_samples']} samples")

    elif args.mode == "evaluate":
        # Run web interface
        create_streamlit_app()

    elif args.mode == "analyze":
        # Analyze results
        evaluator = MOSEvaluator(args.db)
        stats = evaluator.calculate_statistics()

        # Print summary
        print("\n=== MOS Evaluation Results ===")
        print(
            f"Overall MOS: {stats['overall']['mean']:.2f} ± {stats['overall']['std']:.2f}"
        )
        print(f"Total evaluations: {stats['overall']['n_evaluations']}")
        print(f"Number of evaluators: {stats['overall']['n_evaluators']}")

        print("\n=== System Comparison ===")
        for system, data in stats["per_system"].items():
            print(f"\n{system}:")
            print(f"  MOS: {data['mean']:.2f} ± {data['std']:.2f}")
            print(f"  95% CI: [{data['ci_95'][0]:.2f}, {data['ci_95'][1]:.2f}]")

        # Generate plots
        output_dir = Path(args.output)
        evaluator.plot_results(output_dir)

        # Generate report
        evaluator.generate_report(output_dir / "mos_report.html", format="html")

        print(f"\nResults saved to {output_dir}")

    elif args.mode == "web":
        # Run Streamlit app
        os.system(f"streamlit run {__file__} -- --mode evaluate")


if __name__ == "__main__":
    import sys

    if (
        len(sys.argv) > 1
        and sys.argv[1] == "--"
        and sys.argv[2] == "--mode"
        and sys.argv[3] == "evaluate"
    ):
        create_streamlit_app()
    else:
        main()
