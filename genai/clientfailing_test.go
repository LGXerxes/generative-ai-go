package genai

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"google.golang.org/api/option"
)

func TestTwoFunctions(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if testing.Short() {
		t.Skip("skipping live test in -short mode")
	}

	if apiKey == "" {
		t.Skip("set a GEMINI_API_KEY env var to run live tests")
	}

	ctx := context.Background()
	client, err := NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	model := client.GenerativeModel(defaultModel)
	model.Temperature = Ptr[float32](0)
	t.Run("tools", func(t *testing.T) {
		weatherChat := func(t *testing.T, s *Schema, s2 *Schema, fcm FunctionCallingMode) {
			randomTool := &Tool{
				FunctionDeclarations: []*FunctionDeclaration{{
					Name:        "RandomNumber",
					Description: "Get a random number",
					Parameters:  s2,
				}},
			}
			weatherTool := &Tool{
				FunctionDeclarations: []*FunctionDeclaration{{
					Name:        "CurrentWeather",
					Description: "Get the current weather in a given location",
					Parameters:  s,
				}},
			}
			model := client.GenerativeModel(defaultModel)
			model.SetTemperature(0)
			model.Tools = []*Tool{randomTool, weatherTool}
			model.ToolConfig = &ToolConfig{
				FunctionCallingConfig: &FunctionCallingConfig{
					Mode: fcm,
				},
			}
			session := model.StartChat()
			res, err := session.SendMessage(ctx, Text("What is the weather like in New York?"))
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println("response", res.Candidates[0].Content.Parts[0])
			funcalls := res.Candidates[0].FunctionCalls()
			if fcm == FunctionCallingNone {
				if len(funcalls) != 0 {
					t.Fatalf("got %d FunctionCalls, want 0", len(funcalls))
				}
				return
			}
			if len(funcalls) != 1 {
				t.Fatalf("got %d FunctionCalls, want 1", len(funcalls))
			}
			funcall := funcalls[0]
			if g, w := funcall.Name, weatherTool.FunctionDeclarations[0].Name; g != w {
				t.Errorf("FunctionCall.Name: got %q, want %q", g, w)
			}
			locArg, ok := funcall.Args["location"].(string)
			if !ok {
				t.Fatal(`funcall.Args["location"] is not a string`)
			}
			if c := "New York"; !strings.Contains(locArg, c) {
				t.Errorf(`FunctionCall.Args["location"]: got %q, want string containing %q`, locArg, c)
			}
			res, err = session.SendMessage(ctx, FunctionResponse{
				Name: weatherTool.FunctionDeclarations[0].Name,
				Response: map[string]any{
					"weather_there": "cold",
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			checkMatch(t, responseString(res), "(it's|it is|weather) .*cold")
		}
		schema := &Schema{
			Type: TypeObject,
			Properties: map[string]*Schema{
				"location": {
					Type:        TypeString,
					Description: "The city and state, e.g. San Francisco, CA",
				},
				"unit": {
					Type: TypeString,
					Enum: []string{"celsius", "fahrenheit"},
				},
			},
			Required: []string{"location"},
		}
		schemaNumber := &Schema{
			Type: TypeObject,
			Properties: map[string]*Schema{
				"limit": {
					Type:        TypeString,
					Description: "The city and state, e.g. San Francisco, CA",
				},
			},
			Required: []string{"limit"},
		}
		t.Run("direct", func(t *testing.T) {
			weatherChat(t, schema, schemaNumber, FunctionCallingAuto)
		})
		t.Run("none", func(t *testing.T) {
			weatherChat(t, schema, schemaNumber, FunctionCallingNone)
		})
	})
}

func TestTwoFunctionsFlipped(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if testing.Short() {
		t.Skip("skipping live test in -short mode")
	}

	if apiKey == "" {
		t.Skip("set a GEMINI_API_KEY env var to run live tests")
	}

	ctx := context.Background()
	client, err := NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	model := client.GenerativeModel(defaultModel)
	model.Temperature = Ptr[float32](0)
	t.Run("tools", func(t *testing.T) {
		weatherChat := func(t *testing.T, s *Schema, s2 *Schema, fcm FunctionCallingMode) {
			randomTool := &Tool{
				FunctionDeclarations: []*FunctionDeclaration{{
					Name:        "RandomNumber",
					Description: "Get a random number",
					Parameters:  s2,
				}},
			}
			weatherTool := &Tool{
				FunctionDeclarations: []*FunctionDeclaration{{
					Name:        "CurrentWeather",
					Description: "Get the current weather in a given location",
					Parameters:  s,
				}},
			}
			model := client.GenerativeModel(defaultModel)
			model.SetTemperature(0)
			model.Tools = []*Tool{weatherTool, randomTool}
			model.ToolConfig = &ToolConfig{
				FunctionCallingConfig: &FunctionCallingConfig{
					Mode: fcm,
				},
			}
			session := model.StartChat()
			res, err := session.SendMessage(ctx, Text("What is the weather like in New York?"))
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println(res.Candidates[0].Content.Parts[0])
			funcalls := res.Candidates[0].FunctionCalls()
			if fcm == FunctionCallingNone {
				if len(funcalls) != 0 {
					t.Fatalf("got %d FunctionCalls, want 0", len(funcalls))
				}
				return
			}
			if len(funcalls) != 1 {
				t.Fatalf("got %d FunctionCalls, want 1", len(funcalls))
			}
			funcall := funcalls[0]
			if g, w := funcall.Name, weatherTool.FunctionDeclarations[0].Name; g != w {
				t.Errorf("FunctionCall.Name: got %q, want %q", g, w)
			}
			locArg, ok := funcall.Args["location"].(string)
			if !ok {
				t.Fatal(`funcall.Args["location"] is not a string`)
			}
			if c := "New York"; !strings.Contains(locArg, c) {
				t.Errorf(`FunctionCall.Args["location"]: got %q, want string containing %q`, locArg, c)
			}
			res, err = session.SendMessage(ctx, FunctionResponse{
				Name: weatherTool.FunctionDeclarations[0].Name,
				Response: map[string]any{
					"weather_there": "cold",
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			checkMatch(t, responseString(res), "(it's|it is|weather) .*cold")
		}
		schema := &Schema{
			Type: TypeObject,
			Properties: map[string]*Schema{
				"location": {
					Type:        TypeString,
					Description: "The city and state, e.g. San Francisco, CA",
				},
				"unit": {
					Type: TypeString,
					Enum: []string{"celsius", "fahrenheit"},
				},
			},
			Required: []string{"location"},
		}
		schemaNumber := &Schema{
			Type: TypeObject,
			Properties: map[string]*Schema{
				"limit": {
					Type:        TypeString,
					Description: "The city and state, e.g. San Francisco, CA",
				},
			},
			Required: []string{"limit"},
		}
		t.Run("direct", func(t *testing.T) {
			weatherChat(t, schema, schemaNumber, FunctionCallingAuto)
		})
		t.Run("none", func(t *testing.T) {
			weatherChat(t, schema, schemaNumber, FunctionCallingNone)
		})
	})
}
