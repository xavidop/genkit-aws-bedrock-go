// Copyright 2025 Xavier Portilla Edo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// Package main demonstrates advanced schema usage with AWS Bedrock tool calling
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	bedrock "github.com/xavidop/genkit-aws-bedrock-go"
)

// Mock function to process user data
func processUserData(name, email string, age float64, interests []string, subscriptionType string) (string, error) {
	if name == "" || email == "" {
		return "", fmt.Errorf("name and email are required")
	}

	result := map[string]interface{}{
		"processed_user": map[string]interface{}{
			"name":              name,
			"email":             email,
			"age":               age,
			"interests_count":   len(interests),
			"subscription_type": subscriptionType,
			"account_status":    "active",
		},
	}

	jsonResult, _ := json.Marshal(result)
	return string(jsonResult), nil
}

// Mock function to create an order
func createOrder(orderID string, customerName string, itemCount float64, totalAmount float64) (string, error) {
	if orderID == "" || customerName == "" {
		return "", fmt.Errorf("order_id and customer_name are required")
	}

	result := map[string]interface{}{
		"order_created": map[string]interface{}{
			"order_id":           orderID,
			"customer_name":      customerName,
			"item_count":         itemCount,
			"total_amount":       totalAmount,
			"status":             "confirmed",
			"estimated_delivery": "3-5 business days",
		},
	}

	jsonResult, _ := json.Marshal(result)
	return string(jsonResult), nil
}

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

	log.Println("Starting advanced schema tool calling example...")

	// Define Claude 3 model that supports tool calling
	claudeModel := bedrockPlugin.DefineModel(g, bedrock.ModelDefinition{
		Name: "anthropic.claude-3-sonnet-20240229-v1:0",
		Type: "chat",
	}, nil)

	// Define user processing tool with custom schema
	userProcessingTool := genkit.DefineTool(g, "process_user_data",
		"Process and validate user profile data",
		func(ctx *ai.ToolContext, input struct {
			Name             string   `json:"name"`
			Email            string   `json:"email"`
			Age              float64  `json:"age"`
			Interests        []string `json:"interests"`
			SubscriptionType string   `json:"subscription_type"`
		}) (string, error) {
			return processUserData(input.Name, input.Email, input.Age, input.Interests, input.SubscriptionType)
		})

	// Define order creation tool with schema validation
	orderTool := genkit.DefineTool(g, "create_order",
		"Create a new order with customer and item details",
		func(ctx *ai.ToolContext, input struct {
			OrderID      string  `json:"order_id"`
			CustomerName string  `json:"customer_name"`
			ItemCount    float64 `json:"item_count"`
			TotalAmount  float64 `json:"total_amount"`
		}) (string, error) {
			return createOrder(input.OrderID, input.CustomerName, input.ItemCount, input.TotalAmount)
		})

	// Example conversation with advanced tool usage
	prompt := `I need help processing a user profile and creating an order. Here are the details:

User Profile:
- Name: John Doe
- Email: john.doe@example.com  
- Age: 32
- Interests: ["programming", "hiking", "photography"]
- Subscription: premium

Order Details:
- Order ID: ORD-2025-001
- Customer: John Doe
- Item Count: 3
- Total Amount: 149.99

Please process the user data first, then create the order.`

	response, err := genkit.Generate(ctx, g,
		ai.WithModel(claudeModel),
		ai.WithTools(userProcessingTool, orderTool),
		ai.WithMessages(ai.NewUserMessage(ai.NewTextPart(prompt))),
		ai.WithConfig(map[string]interface{}{
			"maxOutputTokens": 1000,
			"temperature":     0.1,
		}),
	)

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	log.Printf("Response: %s", response.Text())
	log.Println("Advanced schema tool calling example completed")
}
