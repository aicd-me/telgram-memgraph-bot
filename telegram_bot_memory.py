import json
import logging
import os
import requests
import asyncio
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from gqlalchemy import Memgraph
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.ERROR)

# Telegram Bot Token from environment variable
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Ollama API endpoint and model name from environment variable
OLLAMA_API = 'http://localhost:11434/api/generate'
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'llama3:8b-instruct-q8_0')

# Memgraph connection
memgraph = Memgraph()

def execute_with_retry(query, params, retries=3, delay=5):
    for i in range(retries):
        try:
            return memgraph.execute(query, params)
        except Exception as e:
            logging.error(f"Attempt {i+1} failed with error: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise

class RateLimiter:
    def __init__(self, max_calls, time_frame):
        self.max_calls = max_calls
        self.time_frame = time_frame
        self.calls = []

    async def wait(self):
        now = time.time()
        self.calls = [call for call in self.calls if call > now - self.time_frame]
        if len(self.calls) >= self.max_calls:
            await asyncio.sleep(self.time_frame - (now - self.calls[0]))
        self.calls.append(now)

# Create a rate limiter for Telegram API calls
telegram_limiter = RateLimiter(max_calls=1, time_frame=3)  # 1 call per 3 seconds

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hello! Happy to talk to you!")

async def send_typing_indicator(context, chat_id):
    try:
        await telegram_limiter.wait()
        
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(1)  # Add a delay to avoid flood control issues
    except Exception as e:
        logging.error(f"Error sending typing indicator: {e}")

async def update_gist_background(user_id, new_message):
    try:
        query = """
        MATCH (u:User {id: $user_id})-[:SENT]->(m:Message)
        RETURN collect(m.`message_text`) AS messages
        """
        result = memgraph.execute_and_fetch(query, {"user_id": user_id})
        messages = next(result)['messages'] + [new_message]

        prompt = "Summarize the following messages:\n" + '\n'.join(messages)
        response = requests.post(OLLAMA_API, json={
            'model': OLLAMA_MODEL_NAME,
            'prompt': prompt
        }, stream=True)
        response.raise_for_status()
        
        # Collect response in parts
        new_gist = ""
        for line in response.iter_lines():
            if line:
                try:
                    line_decoded = json.loads(line.decode('utf-8'))
                    new_gist += line_decoded.get('response', '').strip()
                except json.JSONDecodeError as json_err:
                    logging.error(f"JSON decode error: {json_err}")
                    logging.error(f"Raw response text: {line}")
                    return

        # Store in Memgraph
        store_query = """
        MERGE (u:User {id: $user_id})
        SET u.gist = $gist
        """
        memgraph.execute(store_query, {"user_id": user_id, "gist": new_gist})
    except Exception as e:
        logging.error(f"Failed to update gist: {e}")



def retrieve_conversation_history(entity_id, current_message, is_group=False):
    entity_label = 'Group' if is_group else 'User'
    query = f"""
    MATCH (e:{entity_label} {{id: $entity_id}})-[:SENT]->(m:Message)-[:RESPONDED_WITH]->(r:Message)
    RETURN m.`message_text` AS user_message, r.`message_text` AS ai_response, m.timestamp AS timestamp
    ORDER BY m.timestamp DESC
    LIMIT 20
    """
    try:
        result = memgraph.execute_and_fetch(query, {"entity_id": entity_id})
        history = []
        for record in result:
            history.append({
                "user_message": record['user_message'],
                "ai_response": record['ai_response'],
                "timestamp": record["timestamp"]
            })

        if not history:
            return "No previous conversation history."

        # Use semantic similarity to select the most relevant messages
        vectorizer = TfidfVectorizer()
        corpus = [current_message] + [f"{h['user_message']} {h['ai_response']}" for h in history]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Select top 5 most similar messages
        top_indices = cosine_similarities.argsort()[-5:][::-1]
        relevant_history = [history[i] for i in top_indices]

        # Format the relevant history
        formatted_history = []
        for h in relevant_history:
            formatted_history.append(f"User: {h['user_message']}\nAI: {h['ai_response']}")

        return '\n'.join(formatted_history)
    except Exception as e:
        logging.error(f"Failed to retrieve conversation history: {e}")
        return "Error retrieving conversation history."


def retrieve_conversation_gist(entity_id, is_group=False):
    entity_label = 'Group' if is_group else 'User'
    query = f"""
    MATCH (e:{entity_label} {{id: $entity_id}})
    OPTIONAL MATCH (e)-[:INTERESTED_IN]->(i:Interest)
    RETURN e.gist AS gist, e.name AS name, collect(i.name) AS interests
    """
    try:
        result = memgraph.execute_and_fetch(query, {"entity_id": entity_id})
        entity_data = next(result, {})
        gist = entity_data.get('gist', '')
        profile_info = f"Name: {entity_data.get('name')}, Interests: {', '.join(entity_data.get('interests', []))}"
        return f"{profile_info}\n\nConversation Gist: {gist}" if gist else "No conversation gist available."
    except Exception as e:
        logging.error(f"Failed to retrieve conversation gist: {e}")
        return "Error retrieving conversation gist."


async def extract_entities(user_message, ai_response):
    try:
        semantic_response = requests.post(OLLAMA_API, json={
            'model': OLLAMA_MODEL_NAME,
            'prompt': f"""Extract entities, intents, and sentiments from the following conversation:
            User: {user_message}
            AI: {ai_response}
            
            Please pay special attention to the following entity types:
            - PERSON (for names)
            - AGE
            - GPE (for locations)
            - OCCUPATION
            - INTEREST (for hobbies or topics of interest)
            
            Also, identify intents related to expressing interests, such as 'interest_in_sports' or 'interest_in_technology'.
            """
        }, stream=True)
        semantic_response.raise_for_status()

        combined_responses = ""
        for line in semantic_response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    combined_responses += chunk.get('response', '')
                except json.JSONDecodeError:
                    # If it's not valid JSON, just append the line as is
                    combined_responses += line.decode('utf-8')

        logging.info(f"Combined semantic data response: {combined_responses}")

        # Extract and clean intents, entities, and sentiments
        intents = []
        entities = []
        sentiments = {}

        # Extract intents
        intents_start = combined_responses.find("Intents:")
        intents_end = combined_responses.find("Sentiments:")
        if intents_start != -1 and intents_end != -1:
            intents_str = combined_responses[intents_start+8:intents_end].strip()
            intents = [intent.strip() for intent in intents_str.split("*") if intent.strip()]

        # Extract entities by finding "Entities" section
        entities_start = combined_responses.find("Entities:") + 9
        if entities_start != -1 and intents_start != -1:
            entities_str = combined_responses[entities_start:intents_start].strip()
            for entity in entities_str.split("*"):
                entity = entity.strip()
                if entity:
                    parts = entity.split(":")
                    if len(parts) == 2:
                        name, type_ = parts
                        entities.append({"name": name.strip(), "type": type_.strip()})

        # Extract sentiments
        sentiments_start = combined_responses.find("Sentiments:")
        if sentiments_start != -1:
            sentiments_str = combined_responses[sentiments_start+11:].strip()
            sentiments = {"overall": sentiments_str}

        return intents, entities, sentiments
    except Exception as e:
        logging.error(f"Failed to extract semantic data: {e}")
        return [], [], {}

async def update_user_profile(entity_id, user_message, entities, intents, is_group=False):
    profile_fields = {
        "name": None,
        "age": None,
        "location": None,
        "interests": [],
        "occupation": None
    }

    # Extract profile information from entities
    for entity in entities:
        if entity['type'] == 'PERSON':
            profile_fields['name'] = entity['name']
        elif entity['type'] == 'AGE':
            profile_fields['age'] = entity['name']
        elif entity['type'] == 'GPE':
            profile_fields['location'] = entity['name']
        elif entity['type'] == 'OCCUPATION':
            profile_fields['occupation'] = entity['name']

    # Extract interests from intents
    for intent in intents:
        if intent.lower().startswith('interest_in_'):
            profile_fields['interests'].append(intent.split('_')[-1])

    # Update user or group profile in Memgraph
    entity_label = 'Group' if is_group else 'User'
    query = f"""
    MERGE (e:{entity_label} {{id: $entity_id}})
    SET e.name = COALESCE($name, e.name),
        e.age = COALESCE($age, e.age),
        e.location = COALESCE($location, e.location),
        e.occupation = COALESCE($occupation, e.occupation)
    WITH e
    UNWIND $interests AS interest
    MERGE (i:Interest {{name: interest}})
    MERGE (e)-[:INTERESTED_IN]->(i)
    """
    memgraph.execute(query, {
        "entity_id": entity_id,
        "name": profile_fields['name'],
        "age": profile_fields['age'],
        "location": profile_fields['location'],
        "occupation": profile_fields['occupation'],
        "interests": profile_fields['interests']
    })


async def handle_message(update: Update, context: CallbackContext):
    chat_type = update.effective_chat.type
    is_group = chat_type in ['group', 'supergroup']
    entity_id = update.effective_chat.id if is_group else update.effective_user.id
    user_message = update.message.text

    # Check if the bot is mentioned or the message is a reply to the bot's message in a group chat
    if is_group:
        if not await is_bot_mentioned_or_replied(update, context):
            logging.info("Bot not mentioned or replied to. Ignoring message.")
            return

    try:
        # Send initial typing indicator
        await send_typing_indicator(context, update.effective_chat.id)

        # Retrieve conversation history and gist
        history = retrieve_conversation_history(entity_id, user_message, is_group)
        gist = retrieve_conversation_gist(entity_id, is_group)
        logging.info(f"Retrieved history: {history}")
        logging.info(f"Retrieved gist: {gist}")

        # Combine history and gist in the prompt for LLM context
        combined_context = f"""Here is a summary of the conversation for context: {gist}
        Here are the most relevant previous messages: {history}
        Now respond to the user message below. Use the context to inform your response, but do not explicitly mention or repeat the context.
        Focus on providing a direct and relevant answer to the user's current message.
        User: {user_message}
        AI:"""

        # Ollama inference with streaming
        response = requests.post(OLLAMA_API, json={
            'model': OLLAMA_MODEL_NAME,
            'prompt': combined_context
        }, stream=True)
        response.raise_for_status()

        buffer = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = line.decode('utf-8')
                    json_chunk = json.loads(chunk)
                    buffer += json_chunk.get('response', '')
                    # Send typing indicator every 3 seconds
                    if len(buffer) % 50 == 0:
                        await send_typing_indicator(context, update.effective_chat.id)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to decode JSON chunk: {e}")

        ai_response = buffer
        if update.message:
            await update.message.reply_text(ai_response)

        # Extract entities, intents, and sentiments
        intents, entities, sentiments = await extract_entities(user_message, ai_response)

        # Update user or group profile
        await update_user_profile(entity_id, user_message, entities, intents, is_group)

        # Store conversation and semantic data in Memgraph
        if is_group:
            query = """
            MERGE (g:Group {id: $chat_id})
            CREATE (g_msg:Message {message_text: $user_message, timestamp: timestamp(), type: 'user'})
            CREATE (ai_msg:Message {message_text: $ai_response, timestamp: timestamp(), type: 'ai'})
            MERGE (g)-[:SENT]->(g_msg)
            MERGE (g_msg)-[:RESPONDED_WITH]->(ai_msg)
            FOREACH (intent IN $intents | MERGE (i:Intent {name: intent}) CREATE (g_msg)-[:HAS_INTENT]->(i))
            FOREACH (entity IN $entities | MERGE (e:Entity {name: entity.name, type: entity.type}) CREATE (g_msg)-[:HAS_ENTITY]->(e))
            CREATE (g_msg)-[:HAS_SENTIMENT]->(:Sentiment {sentiment: $sentiment_user})
            CREATE (ai_msg)-[:HAS_SENTIMENT]->(:Sentiment {sentiment: $sentiment_ai})
            WITH g_msg
            MATCH (g_msg)-[:HAS_INTENT]->(i:Intent)
            MERGE (t:Topic {name: i.name})
            MERGE (g_msg)-[:RELATES_TO]->(t)
            """
            memgraph.execute(query, {
                "chat_id": entity_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "intents": intents,
                "entities": entities,
                "sentiment_user": sentiments.get('overall', 'neutral'),
                "sentiment_ai": sentiments.get('overall', 'neutral')
            })
        else:
            query = """
            MERGE (u:User {id: $user_id})
            CREATE (u_msg:Message {message_text: $user_message, timestamp: timestamp(), type: 'user'})
            CREATE (ai_msg:Message {message_text: $ai_response, timestamp: timestamp(), type: 'ai'})
            MERGE (u)-[:SENT]->(u_msg)
            MERGE (u_msg)-[:RESPONDED_WITH]->(ai_msg)
            FOREACH (intent IN $intents | MERGE (i:Intent {name: intent}) CREATE (u_msg)-[:HAS_INTENT]->(i))
            FOREACH (entity IN $entities | MERGE (e:Entity {name: entity.name, type: entity.type}) CREATE (u_msg)-[:HAS_ENTITY]->(e))
            CREATE (u_msg)-[:HAS_SENTIMENT]->(:Sentiment {sentiment: $sentiment_user})
            CREATE (ai_msg)-[:HAS_SENTIMENT]->(:Sentiment {sentiment: $sentiment_ai})
            WITH u_msg
            MATCH (u_msg)-[:HAS_INTENT]->(i:Intent)
            MERGE (t:Topic {name: i.name})
            MERGE (u_msg)-[:RELATES_TO]->(t)
            """
            memgraph.execute(query, {
                "user_id": entity_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "intents": intents,
                "entities": entities,
                "sentiment_user": sentiments.get('overall', 'neutral'),
                "sentiment_ai": sentiments.get('overall', 'neutral')
            })

        # Start background task to update gist
        asyncio.create_task(update_gist_background(entity_id, user_message))

    except Exception as e:
        logging.error(f"Error in handle_message: {e}")
        await update.message.reply_text("I apologize, but I encountered an error while processing your message. Please try again later.")




async def feedback(update: Update, context: CallbackContext):
    feedback_text = update.message.text.lower()
    if feedback_text in ['helpful', 'not helpful']:
        last_message_query = """
        MATCH (u:User {id: $user_id})-[:SENT]->(m:Message)-[:RESPONDED_WITH]->(r:Message)
        WHERE NOT EXISTS((r)-[:HAS_FEEDBACK]->())
        RETURN r ORDER BY r.timestamp DESC LIMIT 1
        """
        result = memgraph.execute_and_fetch(last_message_query, {"user_id": update.effective_user.id})
        last_ai_message = next(result, None)
        if last_ai_message:
            feedback_query = """
            MATCH (m:Message {id: $message_id})
            CREATE (m)-[:HAS_FEEDBACK]->(:Feedback {value: $feedback})
            """
            memgraph.execute(feedback_query, {
                "message_id": last_ai_message['id'],
                "feedback": feedback_text == 'helpful'
            })
            await update.message.reply_text("Thank you for your feedback!")
        else:
            await update.message.reply_text("I couldn't find a recent message to apply feedback to.")
    else:
        await update.message.reply_text("Please provide feedback by saying 'helpful' or 'not helpful'.")

async def is_bot_mentioned_or_replied(update: Update, context: CallbackContext) -> bool:
    message = update.message
    if message is None:
        return False
    
    if message.reply_to_message and message.reply_to_message.from_user.id == context.bot.id:
        return True
    
    if context.bot.username in message.text:
        return True
    
    return False


async def handle_group_message(update: Update, context: CallbackContext):
    if update.message is None or update.message.text is None:
        logging.error("Received an update without a message or text. Skipping processing.")
        return

    # Only proceed if the bot is mentioned or replied to
    if not await is_bot_mentioned_or_replied(update, context):
        logging.info("Bot not mentioned or replied to. Ignoring message.")
        return

    user_message = update.message.text
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    query = """
    MERGE (u:User {id: $user_id})
    CREATE (m:Message {message_text: $message, timestamp: timestamp(), type: 'user', chat_id: $chat_id})
    MERGE (u)-[:SENT]->(m)
    """
    try:
        execute_with_retry(query, {
            "user_id": user_id,
            "message": user_message,
            "chat_id": chat_id
        })
    except Exception as e:
        logging.error(f"Error storing group message: {e}")

    asyncio.create_task(update_gist_background(chat_id, user_message))




async def handle_bot_mention(update: Update, context: CallbackContext):
    user_message = update.message.text
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id  # Ensure user_id captures the user that sent the message in the group

    # Retrieve conversation history and gist
    history = retrieve_conversation_history(chat_id, user_message)
    gist = retrieve_conversation_gist(chat_id)

    # Generate response using Ollama
    combined_context = f"""Here is a summary of the conversation for context: {gist}
    Here are the most relevant previous messages: {history}
    Now respond to the user message below. Use the context to inform your response, but do not explicitly mention or repeat the context.
    Focus on providing a direct and relevant answer to the user's current message.
    User: {user_message}
    AI:"""
    try:
        response = requests.post(OLLAMA_API, json={
            'model': OLLAMA_MODEL_NAME,
            'prompt': combined_context
        }, stream=True)
        response.raise_for_status()

        ai_response = ""
        partial_responses = []

        # Process streaming response
        for i, line in enumerate(response.iter_lines()):
            if line:
                logging.info(f"Interim response line: {line}")
                try:
                    json_line = json.loads(line)
                    partial_responses.append(json_line.get('response', ''))

                    # Send typing indicator asynchronously every 3 lines
                    if i == 0:
                        await(send_typing_indicator(context, chat_id))
                    if json_line.get('done', False):
                        break
                except json.JSONDecodeError as json_err:
                    logging.error(f"JSON decode error: {json_err}")
                    logging.error(f"Raw response text: {line}")
                    await update.message.reply_text("I encountered an error while processing the response. Please try again later.")
                    return

        ai_response = ''.join(partial_responses)
        await update.message.reply_text(ai_response)
        
        # Updated Cypher query
        query = """
        MERGE (u:User {id: $user_id})
        CREATE (m:Message {message_text: $message, timestamp: timestamp(), type: 'bot', chat_id: $chat_id})
        MERGE (u)-[:SENT]->(m)
        """
        memgraph.execute(query, {
            "user_id": context.bot.id,
            "message": ai_response,
            "chat_id": chat_id
        })

    except requests.RequestException as req_err:
        logging.error(f"Request error: {req_err}")
        await update.message.reply_text("I encountered a network error. Please try again later.")
    except Exception as e:
        logging.error(f"Error handling bot mention: {e}")
        await update.message.reply_text("I encountered an error. Please try again later.")



def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CommandHandler("feedback", feedback))
    application.run_polling()

if __name__ == '__main__':
    if TOKEN is None:
        logging.error("TELEGRAM_BOT_TOKEN environment variable not set.")
    else:
        main()
