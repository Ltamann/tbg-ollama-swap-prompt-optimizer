package compat

import "fmt"

type EndpointCapability struct {
	Streaming bool
	Tools     bool
}

type Registry struct {
	endpoints map[EndpointKind]EndpointCapability
}

func NewDefaultRegistry() Registry {
	return Registry{
		endpoints: map[EndpointKind]EndpointCapability{
			EndpointResponses:       {Streaming: true, Tools: true},
			EndpointChatCompletions: {Streaming: true, Tools: true},
			EndpointCompletions:     {Streaming: true, Tools: false},
			EndpointMessages:        {Streaming: true, Tools: false},
			EndpointEmbeddings:      {Streaming: false, Tools: false},
			EndpointImages:          {Streaming: false, Tools: false},
			EndpointAudioSpeech:     {Streaming: true, Tools: false},
			EndpointAudioVoice:      {Streaming: false, Tools: false},
			EndpointAudioTranscribe: {Streaming: false, Tools: false},
			EndpointRerank:          {Streaming: false, Tools: false},
			EndpointInfill:          {Streaming: true, Tools: false},
			EndpointCompletion:      {Streaming: true, Tools: false},
		},
	}
}

func (r Registry) Validate(req CanonicalRequest) error {
	capability, found := r.endpoints[req.Endpoint]
	if !found {
		return fmt.Errorf("endpoint %q is not supported", req.Endpoint)
	}
	if req.HasTools && !capability.Tools {
		return fmt.Errorf("endpoint %q does not support tools", req.Endpoint)
	}
	if req.Stream && !capability.Streaming {
		return fmt.Errorf("endpoint %q does not support streaming", req.Endpoint)
	}
	return nil
}

