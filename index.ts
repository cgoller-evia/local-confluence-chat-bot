import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { ConfluencePagesLoader } from "@langchain/community/document_loaders/web/confluence";
import { Document } from "@langchain/core/documents";
import { AIMessageChunk } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { IterableReadableStream } from "@langchain/core/utils/stream";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { QdrantVectorStore } from "@langchain/qdrant";
import dotenv from 'dotenv';
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import readline from 'readline';

dotenv.config();

const embeddings = new OllamaEmbeddings({
    model: process.env.OLLAMA_EMBEDDING_MODEL,
    baseUrl: process.env.OLLAMA_URL,
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: process.env.QDRANT_URL,
    collectionName: "confluence-chat-bot",
});

const llm = new ChatOllama({
    model: process.env.OLLAMA_CHAT_MODEL,
    temperature: 0,
});

const confluenceLoader = new ConfluencePagesLoader({
    baseUrl: process.env.CONFLUENCE_BASE_URL || '',
    spaceKey: process.env.CONFLUENCE_SPACE_KEY || '',
    personalAccessToken: process.env.CONFLUENCE_PAT,
});

const pdfLoader = new DirectoryLoader("./context", {
    ".pdf": (path: string) => new PDFLoader(path),
});

async function buildIndex(loader: ConfluencePagesLoader | DirectoryLoader) {
    console.log("Loading documents...");
    const docs = await loader.load();
    console.log(`Loaded ${docs.length} documents.`);

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1500,
        chunkOverlap: 50
    });
    console.log("Splitting documents...");
    const splitDocs = await textSplitter.splitDocuments(docs);
    console.log("Adding documents to vector store...");
    await vectorStore.addDocuments(splitDocs);
    console.log("Index built successfully.\n");
}

async function startChat() {
    const template = `Answer the users QUESTION using the DOCUMENT text above.
        Keep your answer ground in the facts of the DOCUMENT.
        If the DOCUMENT does not contain the facts to answer the QUESTION, just say that you don't know.
        QUESTION: {question}
        DOCUMENT: {context}
        ANSWER:
    `;
    const promptTemplate = ChatPromptTemplate.fromTemplate(template);
    
    const InputStateAnnotation = Annotation.Root({
        question: Annotation<string>,
    });

    const StateAnnotation = Annotation.Root({
        question: Annotation<string>,
        context: Annotation<Document[]>,
        answer: Annotation<IterableReadableStream<AIMessageChunk>>,
    });

    const retrieve = async (state: typeof InputStateAnnotation.State) => {
        const retrievedDocs = await vectorStore.similaritySearch(state.question, 4)
        return { context: retrievedDocs };
    };

    const generate = async (state: typeof StateAnnotation.State) => {
        const docsContent = state.context.map(doc => `${doc.metadata.title}\n ${doc.pageContent}`).join("\n");
        const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
        const response = await llm.stream(messages);
        return { answer: response };
    };

    const graph = new StateGraph(StateAnnotation)
        .addNode("retrieve", retrieve)
        .addNode("generate", generate)
        .addEdge("__start__", "retrieve")
        .addEdge("retrieve", "generate")
        .addEdge("generate", "__end__")
        .compile();

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: "You: "
    });

    console.log("ChatBot is ready! Type your questions, or type 'bye' to exit.\n");
    rl.prompt();

    rl.on('line', async (line) => {
        const input = line.trim();
        if (input.toLowerCase() === 'bye') {
            console.log("Goodbye!");
            rl.close();
            process.exit(0);
        }
        try {
            const response = await graph.invoke({ question: input});
            for await (const chunk of response.answer) {
                process.stdout.write(chunk.content as string);            
            }            
            console.log("\n---------------------------------------------------------");
        } catch (error) {
            console.error("Error processing query:", error);
        }
        rl.prompt();
    });
}

async function main() {
    try {
        //await buildIndex(confluenceLoader);
        await startChat();
    } catch (error) {
        console.error("Error in main:", error);
    }
}

main();
