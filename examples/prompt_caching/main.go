package main

import (
	"context"
	_ "embed"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
	"log"
)

// A prompt has to be big enough for being cached
// https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models
//
//go:embed sysprompt.md
var sysprompt string

func main() {
	ctx := context.Background()

	// Initialize Bedrock plugin
	bedrockPlugin := &bedrock.Bedrock{
		Region: "us-east-1",
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(bedrockPlugin),
	)

	// Define Claude 3.7 Sonnet model
	claudeModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
		Type: "chat",
	}, nil)

	// User prompts
	inputs := []string{
		"Write a four-line poem about lost love",
		"Write a four-line poem about autumn",
	}

	// call several times in a row to verify caching works
	for _, input := range inputs {
		response, err := genkit.Generate(ctx, g,
			ai.WithModel(claudeModel),
			ai.WithMessages(
				ai.NewSystemMessage(
					ai.NewTextPart(sysprompt),
					bedrock.NewCachePointPart(), // add a cache point after the system prompt
				),
				ai.NewUserTextMessage(input),
			),
		)
		if err != nil {
			log.Fatal(err)
		}
		log.Println("Request:", input)
		log.Printf("Response:\n%s\n", response.Text())
		log.Println("Tokens read from cache:", response.Usage.CachedContentTokens)
	}
}
