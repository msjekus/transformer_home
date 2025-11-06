import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import threading

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-2 Чат")
        self.root.geometry("750x750")
        self.history_file = "chat_history.txt"
        self.create_widgets()
        self.load_model()

    def create_widgets(self):
        chat_label = tk.Label(self.root, text="Історія чату:", font=("Arial", 11, "bold"))
        chat_label.pack(anchor='w', padx=10)

        self.chat_history = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Arial", 10),
            height=20
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.chat_history.config(state=tk.DISABLED)

        input_label = tk.Label(self.root, text="Ваш запит:", font=("Arial", 14, "bold"))
        input_label.pack(anchor='w', padx=10, pady=(10, 0))

        self.input_text = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Arial", 14),
            height=4
        )
        self.input_text.pack(fill=tk.X, padx=10, pady=5)

        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.send_button = tk.Button(
            button_frame,
            text="Відправити",
            font=("Arial", 11, "bold"),
            command=self.send_message,
            bg='#4CAF50',
            fg='white',
            height=2
        )
        self.send_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        clear_button = tk.Button(
            button_frame,
            text="Очистити",
            font=("Arial", 11, "bold"),
            command=self.clear_chat,
            bg='#f44336',
            fg='white',
            height=2
        )
        clear_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

    def load_model(self):
        self.add_to_chat("СИСТЕМА", "Завантаження моделі...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.add_to_chat("СИСТЕМА", "Модель завантажена! Готовий до роботи.")

    def send_message(self):
        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Увага", "Введіть запит!")
            return

        self.input_text.delete("1.0", tk.END)
        self.add_to_chat("ВИ", prompt)
        thread = threading.Thread(target=self.generate_response, args=(prompt,), daemon=True)
        thread.start()

    def generate_response(self, prompt):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            response = generated_text[len(prompt):].strip()
            if not response:
                response = generated_text

            self.add_to_chat("GPT-2", response)

        except Exception as e:
            self.add_to_chat("ПОМИЛКА", str(e))
        finally:
            self.send_button.config(state=tk.NORMAL, text="Відправити")

    def add_to_chat(self, sender, message):
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, f"[{timestamp}] {sender}:\n{message}\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {sender}:\n{message}\n\n")
        except Exception as e:
            print(f"Помилка збереження: {e}")

    def clear_chat(self):

        if messagebox.askyesno("Підтвердження", "Очистити історію чату?"):
            self.chat_history.config(state=tk.NORMAL)
            self.chat_history.delete("1.0", tk.END)
            self.chat_history.config(state=tk.DISABLED)
            try:
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    f.write("")
            except:
                pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
