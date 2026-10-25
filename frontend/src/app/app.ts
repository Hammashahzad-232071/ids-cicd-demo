import { Component, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FEATURES, SAMPLES } from './model-schema';

interface Result {
  prediction: string;
  confidence: number;
  is_attack: boolean;
  probabilities: Record<string, number>;
}

@Component({
  selector: 'app-root',
  imports: [FormsModule, DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  features = FEATURES;
  sampleNames = Object.keys(SAMPLES);
  values: Record<string, number> = { ...SAMPLES['BENIGN'] };
  result = signal<Result | null>(null);
  message = signal('');
  loading = signal(false);
  isStaticHost = window.location.hostname !== 'localhost';

  loadSample(name: string) {
    this.values = { ...SAMPLES[name] };
    this.result.set(null);
    this.message.set('');
  }

  async predict() {
    this.loading.set(true);
    this.result.set(null);
    this.message.set('');
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.values),
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      this.result.set(await res.json());
    } catch {
      this.message.set(
        'Backend not reachable. Start Flask locally (python backend/app.py) and run ng serve with the proxy.'
      );
    } finally {
      this.loading.set(false);
    }
  }

  probs() {
    const r = this.result();
    return r ? Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]) : [];
  }
}
