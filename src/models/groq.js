import Groq from 'groq-sdk';
import { getKey, hasKey } from '../utils/keys.js';

export class GroqWrapper {
    constructor(model_name) {
        this.model_name = model_name || "gemma2-9b-it";
        this.groq = new Groq({ apiKey: getKey('GROQ_API_KEY') });
    }

    async sendRequest(turns, systemMessage, stop_seq = '***') {
        let messages = [{ 'role': 'system', 'content': systemMessage }].concat(turns);
        let res = null;
        try {
            console.log('Awaiting Groq API response...');
            let completion = await this.groq.chat.completions.create({
                model: this.model_name,
                messages: messages,
                stop: stop_seq,
            });
            if (completion.choices[0].finish_reason === 'length') {
                throw new Error('Context length exceeded');
            }
            console.log('Received.');
            res = completion.choices[0].message.content;
        } catch (err) {
            if ((err.message === 'Context length exceeded' || err.code === 'context_length_exceeded') && turns.length > 1) {
                console.log('Context length exceeded, trying again with shorter context.');
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            } else {
                console.log(err);
                res = 'My brain disconnected, try again.';
            }
        }
        return res;
    }

    async embed(text) {
        // Note: As of my last update, Groq doesn't provide a direct embedding API.
        // This method is included for API compatibility, but it will throw an error.
        throw new Error('Embedding is not currently supported by the Groq API.');
    }
}